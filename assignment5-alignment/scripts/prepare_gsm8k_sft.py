import argparse
import json


def extract_gsm8k_final_answer(answer):
    """从 GSM8K answer 字段中提取 #### 后的最终答案。"""
    return answer.split("####")[-1].strip() if "####" in answer else answer.strip()


def format_gsm8k_sft_response(answer):
    """把 GSM8K 解题过程包装成 r1_zero.prompt 对应的 response。"""
    if "####" in answer:
        reasoning, final_answer = answer.rsplit("####", 1)
        reasoning = reasoning.strip()
        final_answer = final_answer.strip()
    else:
        reasoning = answer.strip()
        final_answer = answer.strip()
    return f"{reasoning}\n</think><answer>{final_answer}</answer>"


def main():
    parser = argparse.ArgumentParser(description="Convert GSM8K jsonl to r1_zero SFT format.")
    parser.add_argument("--input_path", default="data/gsm8k/train.jsonl")
    parser.add_argument("--output_path", default="data/gsm8k/train_sft_r1_zero.jsonl")
    parser.add_argument("--prompt_path", default="cs336_alignment/prompts/r1_zero.prompt")
    args = parser.parse_args()

    with open(args.prompt_path) as f:
        prompt_template = f.read().strip()

    num_examples = 0
    with open(args.input_path) as fin, open(args.output_path, "w") as fout:
        for line in fin:
            item = json.loads(line)
            output_item = {
                "prompt": prompt_template.replace("{question}", item["question"]),
                "response": format_gsm8k_sft_response(item["answer"]),
                "ground_truth": extract_gsm8k_final_answer(item["answer"]),
            }
            fout.write(json.dumps(output_item, ensure_ascii=False) + "\n")
            num_examples += 1

    print(f"Wrote {num_examples} examples to {args.output_path}")


if __name__ == "__main__":
    main()
