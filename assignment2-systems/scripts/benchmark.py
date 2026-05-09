from pickletools import optimize
import timeit
from tracemalloc import start
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'cs336-basics'))
import pandas as pd
from statistics import mean, stdev
from cs336_basics.model import BasicsTransformerLM
import argparse
from typing import Tuple, Optional
import json
from pathlib import Path
from contextlib import nullcontext
model_configs = [
    # {"size": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    # {"size": "medium","d_model": 1024,"d_ff": 4096,"num_layers": 24,"num_heads": 16,},
    # {"size": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    # {"size": "xl", "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    {"size": "2.7B", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
]

#Hyperarameters
vocab_size = 10_000
context_length = 512
batch_size = 1
rope_theta = 10000.0
warmup_steps = 5
timing_steps = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

def profile_one_train_step(model, x, y, vocab_size, tag: str = ''):
    '''
    仅执行一次训练步,并分别对 forward/ backward / optimizer step
    开启独立的内存历史记录与导出快照(c++&python)

    '''
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # ------- Forward --------
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(
        max_entries=1_000_000, context='all', stacks='all'
    )
    torch.cuda.reset_peak_memory_stats()
    out = model(x)                      # 注意：不加 no_grad，保留激活供 backward
    torch.cuda.synchronize()
    fwd_alloc = torch.cuda.max_memory_allocated()
    fwd_res   = torch.cuda.max_memory_reserved()
    torch.cuda.memory._dump_snapshot(f"memory_forward{('_'+tag) if tag else ''}.pickle")
    torch.cuda.memory._record_memory_history(enable=None)


    # ------- Backward ---------
    loss = criterion(out.view(-1, vocab_size), y.view(-1))
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(
        max_entries=1_000_000, context='all', stacks='all'
    )
    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    torch.cuda.synchronize()
    bwd_alloc = torch.cuda.max_memory_allocated()
    bwd_res   = torch.cuda.max_memory_reserved()
    torch.cuda.memory._dump_snapshot(f"memory_backward{('_' + tag) if tag else ''}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    del loss, out
    torch.cuda.synchronize()
    torch.cuda.synchronize()

    # --------Optimizer step---------
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(
        max_entries=1_000_000, context='all', stacks='all'
    )
    torch.cuda.reset_peak_memory_stats()
    optimizer.step()                     # 注意：第一次 step 会分配优化器状态
    torch.cuda.synchronize()
    opt_alloc = torch.cuda.max_memory_allocated()
    opt_res   = torch.cuda.max_memory_reserved()
    torch.cuda.memory._dump_snapshot(f"memory_optim{('_' + tag) if tag else ''}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

    # 方便在控制台也看到峰值（字节）
    print(
        f"[PEAK] forward  alloc={fwd_alloc/1e9:.3f} GB, reserved={fwd_res/1e9:.3f} GB\n"
        f"[PEAK] backward alloc={bwd_alloc/1e9:.3f} GB, reserved={bwd_res/1e9:.3f} GB\n"
        f"[PEAK] optim    alloc={opt_alloc/1e9:.3f} GB, reserved={opt_res/1e9:.3f} GB"
    )