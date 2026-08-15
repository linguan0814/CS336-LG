#!/usr/bin/env python3
"""
Text generation script for Transformer language model.

This script implements decoding functionality including temperature scaling,
top-p (nucleus) sampling, and text generation from trained models.
"""

import argparse
import torch
from typing import Optional
import sys
import os

# Add the project root to Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.model.transformer import transformer_lm
from cs336_basics.tokenizer import Tokenizer


DEFAULT_MODEL_CONFIG = {
    'vocab_size': 10000,
    'context_length': 256,
    'num_layers': 4,
    'd_model': 512,
    'num_heads': 16,
    'rope_theta': 10000.0,
    'd_ff': 1344,
}


def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling and softmax to logits.
    
    Args:
        logits: Raw logits tensor of shape (..., vocab_size)
        temperature: Temperature parameter for scaling. Lower values make distribution more peaked.
    
    Returns:
        Softmax probabilities with temperature scaling
    """
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=-1)


def top_p_sampling(probabilities: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to probability distribution.
    
    Args:
        probabilities: Probability distribution of shape (..., vocab_size)
        p: Cumulative probability threshold for nucleus sampling
    
    Returns:
        Modified probability distribution with low-probability tokens masked out
    """
    if not 0 < p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")
    if p == 1:
        return probabilities

    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    remove_mask = cumulative_probs > p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False
    mask = ~remove_mask

    filtered_probs = sorted_probs * mask.float()
    filtered_probs = filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True)

    output_probs = torch.zeros_like(probabilities)
    output_probs.scatter_(-1, sorted_indices, filtered_probs)
    return output_probs


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cpu",
    eos_token: Optional[str] = "<|endoftext|>"
) -> str:
    """
    Generate text from a trained language model.
    
    Args:
        model: Trained transformer language model
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Input prompt string
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (lower = more deterministic)
        top_p: Top-p threshold for nucleus sampling
        device: Device to run generation on
        eos_token: End-of-sequence token
    
    Returns:
        Generated text string
    """
    model.eval()

    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        raise ValueError("prompt must encode to at least one token")

    generated_tokens = prompt_tokens.copy()

    with torch.no_grad():
        for _ in range(max_tokens):
            context_tokens = generated_tokens[-model.context_length:]
            input_ids = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_ids)

            next_token_logits = logits[0, -1, :]
            probabilities = softmax_with_temperature(next_token_logits, temperature)
            probabilities = top_p_sampling(probabilities, top_p)

            next_token = torch.multinomial(probabilities, num_samples=1).item()
            generated_tokens.append(next_token)

            if eos_token is not None:
                decoded_token = tokenizer.decode([next_token])
                if decoded_token == eos_token:
                    break

    return tokenizer.decode(generated_tokens)


def get_checkpoint_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        if 'model_state' in checkpoint:
            return checkpoint['model_state']
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
    return checkpoint


def build_model_config(args, checkpoint):
    config = DEFAULT_MODEL_CONFIG.copy()
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        config.update(checkpoint['model_config'])
    else:
        print("Warning: checkpoint has no model_config; using CLI/default model arguments.")

    config.update({
        'vocab_size': args.vocab_size,
        'context_length': args.context_len,
        'num_layers': args.num_layers,
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'rope_theta': args.rope_theta,
        'd_ff': args.d_ff,
    })
    return config


def load_model_and_tokenizer(args, device: str = "cpu"):
    """
    Load a trained model and tokenizer from checkpoint and vocab files.
    
    Args:
        args: Parsed command line arguments
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=args.special_tokens,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = build_model_config(args, checkpoint)

    model = transformer_lm(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        rope_theta=config['rope_theta'],
        d_ff=config['d_ff'],
    ).to(device)

    model.load_state_dict(get_checkpoint_model_state(checkpoint))
    model.eval()

    print(f"Loaded checkpoint iteration: {checkpoint.get('iteration', 'unknown') if isinstance(checkpoint, dict) else 'unknown'}")
    print(f"Model config: {config}")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Generate text with a trained Transformer language model')
    
    # Model and tokenizer paths
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True, help='Path to tokenizer vocabulary file')
    parser.add_argument('--merges', type=str, required=True, help='Path to tokenizer merges file')

    # Model arguments. These are used for old checkpoints that do not contain model_config.
    parser.add_argument('--vocab_size', type=int, default=10000, help='Size of vocabulary')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--d_ff', type=int, default=1344, help='FFN dimension')
    parser.add_argument('--context_len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='Input prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=256, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling (lower = more deterministic)')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p threshold for nucleus sampling')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, cuda, mps')
    
    # End-of-sequence token
    parser.add_argument('--eos_token', type=str, default='<|endoftext|>', help='End-of-sequence token')
    parser.add_argument(
        '--special_tokens',
        type=str,
        nargs='*',
        default=['<|endoftext|>', '<|im_start|>', '<|im_end|>'],
        help='Special tokens to register with the tokenizer',
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    try:
        model, tokenizer = load_model_and_tokenizer(args, device=device)
        print("Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Generate text
    print(f"\nGenerating {args.num_samples} sample(s) with prompt: '{args.prompt}'")
    print(f"Parameters: max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p}")
    print("-" * 80)
    
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\nSample {i+1}:")
            print("-" * 40)
        
        try:
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                eos_token=args.eos_token
            )
            
            print(generated_text)
            print("\n" + "=" * 80)
            
        except Exception as e:
            print(f"Error during generation: {e}")
            break


if __name__ == '__main__':
    main()
