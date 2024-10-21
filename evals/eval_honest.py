import os
import sys
import json
import argparse
import time
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import torch

# Adjust the import path if necessary
# Assuming eval_honest.py is in the same directory as server
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.steering_api_functions import create_steerable_model_honesty

def evaluate_honesty(
    model_info: Dict[str, Any],
    inputs: List[str],
    control_setting: Dict[str, float],
    max_new_tokens: int = 128,
) -> List[str]:
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    control_vector = model_info["control_vector"]
    rep_control_pipeline = model_info["rep_control_pipeline"]
    layer_id = model_info["layer_id"]

    coeff = control_setting.get("honesty", 0)
    activations = {}
    for layer in layer_id:
        activations[layer] = torch.tensor(
            coeff * control_vector.directions[layer] * control_vector.direction_signs[layer]
        ).to(model.device).half()

    outputs = rep_control_pipeline(
        inputs,
        activations=activations if coeff != 0 else None,
        batch_size=4,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    return [
        output[0]['generated_text'].replace(input_text, "")
        for output, input_text in zip(outputs, inputs)
    ]

def main(
    model_label: str,
    data_path: str,
    num_samples: int,
    control_settings_list: List[Dict[str, float]],
):
    # Create the steerable model with the honesty control vector
    model_info = create_steerable_model_honesty(
        model_label=model_label,
        data_path=data_path,
        user_tag="USER:",
        asst_tag="ASSISTANT:",
    )

    # Prepare test inputs
    dataset = model_info["control_vector"].dataset  # Access the dataset used
    test_inputs = dataset['test']['data'][:num_samples]

    results = []

    for control_setting in control_settings_list:
        print(f"Running evaluation with control settings: {control_setting}")

        outputs = evaluate_honesty(
            model_info=model_info,
            inputs=test_inputs,
            control_setting=control_setting,
            max_new_tokens=128,
        )

        for input_text, output_text in zip(test_inputs, outputs):
            results.append({
                "input": input_text,
                "output": output_text,
                "control_setting": control_setting,
            })

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"honesty_eval_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate honesty control")
    parser.add_argument(
        "--model_label",
        type=str,
        default="aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored",
        help="Model label",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/facts/facts_true_false.csv",
        help="Path to the dataset CSV file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate",
    )

    args = parser.parse_args()

    # Define control settings to evaluate
    control_settings_list = [
        {},  # No control
        {"honesty": 2},  # Increase honesty
        {"honesty": -2},  # Decrease honesty
    ]

    main(
        model_label=args.model_label,
        data_path=args.data_path,
        num_samples=args.num_samples,
        control_settings_list=control_settings_list,
    )
