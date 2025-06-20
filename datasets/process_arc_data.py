import argparse
import datasets
import json
import os
from typing import Dict


def build_hf_dataset(prefix: str):
    data = {}
    suffixes = ["challenges", "solutions"]
    for suffix in suffixes:
        with open(f"{prefix}_{suffix}.json", "r") as f:
            file_data = json.load(f)
            for key in file_data:
                if suffix == "challenges":
                    data[key] = data.get(key, {})
                    data[key]["examples"] = file_data[key]["train"]
                    data[key]["predict"] = file_data[key]["test"][0]
                elif suffix == "solutions":
                    data[key]["predict"]["output"] = file_data[key][0]
    return datasets.Dataset.from_list([*data.values()])


def construct_prompt(sample: Dict, pred_pair: Dict, user_template: str):
    example_pairs = sample.pop("examples")
    prompt = f"{user_template}\n\n"
    for example_idx, example in enumerate(example_pairs):
        prompt += f"## Example {example_idx + 1}\n\n"
        prompt += f"### Input\n\n{example['input']}\n\n"
        prompt += f"### Output\n\n{example['output']}\n\n"
    prompt += f"### Test case\n\n### Input\n\n{pred_pair['input']}\n\n"
    return prompt


def make_map_fn(split: str, data_source: str, template_path: str):
    def process_fn(sample: dict, idx: int):
        pred_pair = sample.pop("predict")
        with open(f"{template_path}/user.txt", "r") as file:
            user_template = file.read()
        with open(f"{template_path}/system.txt", "r") as file:
            system_prompt = file.read()
        user_prompt = construct_prompt(sample, pred_pair, user_template)
        return {
            "data_source": data_source,
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": pred_pair["output"],
            },
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "extra_info": {
                "split": split,
                "index": idx,
            },
        }
    return process_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default="../../datasets/arc-2025-parquet")
    parser.add_argument("--raw-data-dir", default="./raw-data")
    parser.add_argument("--data-source", default="ARC 2025 Data")
    parser.add_argument("--template-path", default="./templates/default-templates")
    args = parser.parse_args()
    for subset in ["train", "test"]:
        parquet_path = os.path.join(args.save_dir, f"{subset}.parquet")
        suffix = "training" if subset == "train" else "evaluation"
        mapping = make_map_fn(subset, args.data_source, args.template_path)
        dataset = build_hf_dataset(f"{args.raw_data_dir}/arc-agi_{suffix}")
        dataset = dataset.map(mapping, with_indices=True)
        dataset.to_parquet(parquet_path)


if __name__ == "__main__":
    main()