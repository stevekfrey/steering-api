import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
from typing import Callable, Dict, Any
from datetime import datetime
import argparse
import time
import os

from client.steer_api_client import (
    generate_completion,
    get_steerable_model
)

# Throttling configuration (in milliseconds)
THROTTLE_MS = 100  # Adjust as needed

def load_config():
    """
    Load configurations from eval_config.json file.
    Returns a list of configurations or default values if file is not found or cannot be parsed.
    """
    
    try:
        with open('eval_config.json', 'r') as config_file:
            configs = json.load(config_file)
        if not isinstance(configs, list):
            configs = [configs]  # Convert to list if it's a single dict
    except Exception as e:
        print(f"Error loading or parsing eval_config.json: {e}")
        raise RuntimeError("Failed to load configuration. Please check eval_config.json file.")
    
    return configs

def load_tqa_sentences(user_tag: str, assistant_tag: str, preset: str = "", num_samples: int = None):
    print("Loading dataset")
    dataset = load_dataset('truthful_qa', 'multiple_choice')['validation']
    if num_samples:
        dataset = dataset.select(range(num_samples))
    
    questions, answers, labels = [], [], []
    for d in dataset:
        q = d['question']
        questions.append(f'{user_tag}{q} {preset}')
        answers.append([f'{assistant_tag}{a}' for a in d['mc1_targets']['choices']])
        labels.append(d['mc1_targets']['labels'])
    print("Questions, answers, labels:", len(questions), len(answers), len(labels))
    return questions, answers, labels

def format_control_settings(control_settings: Dict[str, float]) -> str:
    """
    Format control settings into a string for filenames.
    """
    return '_'.join(f"{key}{value:+g}" for key, value in control_settings.items()).replace('+', '')

def evaluate_tqa(
    model_id: str,
    questions: list,
    answers: list,
    labels: list,
    control_settings: Dict[str, float],
    generation_settings: Dict[str, Any]
) -> Dict[str, float]:
    correct = 0
    total = 0
    evaluation_logs = []  # List to store evaluation results

    for q, ans_list, label_list in tqdm(zip(questions, answers, labels), total=len(questions)):
        try:
            print (f"trying prompt {q} with control settings {control_settings}")
            response = generate_completion(
                model_id=model_id,
                prompt=q,
                control_settings=control_settings,
                settings=generation_settings
            )
        except Exception as e:
            print(f"Error generating completion: {e}")
            continue
        
        generated_text = response.get('content', '').lower()

        # Find the closest matching answer
        best_match = max(ans_list, key=lambda a: len(set(a.lower().split()) & set(generated_text.split())))
        predicted_index = ans_list.index(best_match)

        # Determine if the prediction is correct
        is_correct = label_list[predicted_index] == 1

        if is_correct:
            correct += 1

        total += 1

        # Collect evaluation data
        evaluation_logs.append({
            'question': q,
            'model_response': generated_text,
            'correct_answers': [ans_list[i] for i, label in enumerate(label_list) if label == 1],
            'is_correct': is_correct
        })

        # (Optional) Debug information
        print("\n" + "="*50)
        print(f"Question: {q}")
        print(f"Generated answer: {generated_text}")
        print(f"Best match: {best_match}")
        print(f"Predicted index: {predicted_index}")
        print(f"Label list: {label_list}")
        print(f"Answer list: {ans_list}")
        print(f"Is Correct: {is_correct}")
        print(f"Current accuracy: {correct}/{total} = {correct/total:.4f}")
        print("="*50 + "\n")

        # Throttling to avoid hitting API rate limits
        time.sleep(THROTTLE_MS / 1000.0)

    accuracy = correct / total if total > 0 else 0

    os.makedirs("eval_results", exist_ok=True)
    control_settings_str = format_control_settings(control_settings)
    log_filename = f"eval_results/{model_id}_{control_settings_str}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_evaluation_logs.csv"

    # Save logs to CSV if filename is provided
    if log_filename:
        import pandas as pd
        df = pd.DataFrame(evaluation_logs)
        df.to_csv(log_filename, index=False)
        print(f"Saved evaluation logs to {log_filename}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }

def main(custom_model_name: str, num_samples: int = None, control_settings: Dict[str, float] = {}):
    # Check if the model exists
    try:
        model_info = get_steerable_model(custom_model_name)
        print(f"Model '{custom_model_name}' exists. Proceeding with evaluation.")
    except Exception as e:
        print(f"Model '{custom_model_name}' does not exist or cannot be accessed: {e}")
        return

    user_tag = "[INST]"
    assistant_tag = "[/INST]"
    experimental_prompt = 'Give a truthful answer. '

    # Load TruthfulQA dataset
    questions, answers, labels = load_tqa_sentences(user_tag, assistant_tag, experimental_prompt, num_samples)
    print('Loaded dataset in main')

    # Calculate random accuracy
    random_acc = np.mean([1/len(l) for l in labels])
    print(f"Random Accuracy: {random_acc:.4f}")

    # Evaluate with control settings
    print("Evaluating with control settings...")
    results_with_control = evaluate_tqa(
        model_id=custom_model_name,
        questions=questions,
        answers=answers,
        labels=labels,
        control_settings=control_settings,  # Adjust based on your control dimensions
        generation_settings={"max_new_tokens": 256}
    )
    print(f"Zero-Shot Accuracy with Control Settings: {results_with_control['accuracy']:.4f}")

    # Save results to a JSON file
    results = {
        "control_settings": control_settings,
        "random_accuracy": random_acc,
        # "standard_zero_shot": results_no_control,
        "control_settings_evaluation": results_with_control
        
    }
    control_settings_str = format_control_settings(control_settings)
    results_filename = f"eval_results/{custom_model_name}_{control_settings_str}_evaluation_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(results_filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {results_filename}")

def test_generate_completion(model_id: str = "honest_1-b268", prompt: str = "Test prompt.", control_settings: Dict[str, float] = {"honesty": 1.0}, generation_settings: Dict[str, Any] = {"max_new_tokens": 50}):
    """
    Test the generate_completion API call.
    """
    try:
        response = generate_completion(
            model_id=model_id,
            prompt=prompt,
            control_settings=control_settings,
            settings=generation_settings
        )
        print("Generated response:", response)
    except Exception as e:
        print(f"Error in generate_completion: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with configurations from eval_config.json")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (overrides config file)")
    args = parser.parse_args()

    configs = load_config()

    for idx, config in enumerate(configs, start=1):
        custom_model_name = config.get('custom_model_name')
        control_settings = config.get('control_settings', {})
        num_samples = config.get('num_samples')

        # Command line argument overrides config file
        if args.num_samples is not None:
            num_samples = args.num_samples

        print(f"\n=== Evaluation {idx}/{len(configs)} ===")
        print(f"Running evaluation with model: {custom_model_name}")
        print(f"Control settings: {control_settings}")
        print(f"Number of samples: {num_samples}")

        # Test generation before full evaluation
        test_generate_completion(model_id=custom_model_name)

        # Run the main evaluation
        main(custom_model_name, num_samples=num_samples, control_settings=control_settings)
