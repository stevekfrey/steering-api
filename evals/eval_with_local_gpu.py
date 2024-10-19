import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
from typing import Callable, Dict, Any
from steer_templates import BASE_MODEL_NAME
from datetime import datetime
import argparse

from steering_api_functions import load_model, generate_completion_response, get_steerable_model
# from steer_api_client import generate_completion, get_steerable_model 


def load_tqa_sentences(user_tag: str, assistant_tag: str, preset: str = "", num_samples: int = None):
    print("loading dataset")
    dataset = load_dataset('truthful_qa', 'multiple_choice')['validation']
    if num_samples:
        dataset = dataset.select(range(num_samples))
    
    questions, answers, labels = [], [], []
    for d in dataset:
        q = d['question']
        questions.append(f'{user_tag}{q} {preset}')
        answers.append([f'{assistant_tag}{a}' for a in d['mc1_targets']['choices']])
        labels.append(d['mc1_targets']['labels'])
    print ("questions, answers, labels: ", len(questions), len(answers), len(labels))
    print (questions, answers, labels)
    return questions, answers, labels

def evaluate_tqa(
    generate_func: Callable,
    model: Any,
    tokenizer: Any,
    questions: list,
    answers: list,
    labels: list,
    model_name: str,
    control_settings: Dict[str, float],
    generation_settings: Dict[str, Any]
) -> Dict[str, float]:
    correct = 0
    total = 0
    evaluation_logs = []  # List to store evaluation results

    # Before the evaluation loop
    steerable_model = get_steerable_model(model_name)
    if steerable_model:
        control_vectors = steerable_model['control_vectors']

    for q, ans_list, label_list in tqdm(zip(questions, answers, labels), total=len(questions)):
        
        response = generate_func(
            model_name_request=model_name,
            prompt=q,
            control_settings=control_settings,
            generation_settings=generation_settings,
            model=model,
            tokenizer=tokenizer
        )
        
        generated_text = response['content'].lower()

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

    accuracy = correct / total if total > 0 else 0

    log_filename = f"{model_name}_{str(control_settings.keys())}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_evaluation_logs.csv"

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


def generate_with_api(
    model_name_request,
    prompt,
    control_settings,
    generation_settings,
    model=None,  # Not used in API call
    tokenizer=None  # Not used in API call
):
    """
    Generates a completion using the API client function.

    Args:
        model_name_request (str): The name/ID of the model to use.
        prompt (str): The input prompt.
        control_settings (dict): Settings for control dimensions.
        generation_settings (dict): Settings for text generation.
        model: Ignored (kept for compatibility with local generation function).
        tokenizer: Ignored (kept for compatibility with local generation function).

    Returns:
        dict: A dictionary containing the generated response and metadata.
    """
    try:
        # Call the API client function
        content = generate_completion(
            model_id=model_name_request,
            prompt=prompt,
            control_settings=control_settings,
            settings=generation_settings
        )

        # Format the response to match the expected structure
        response = {
            'id': 'api_generated',  # You might want to generate a unique ID here
            'object': 'text_completion',
            'created': None,  # The API doesn't return a creation timestamp
            'model': model_name_request,
            'content': content
        }

        return response

    except Exception as e:
        # Log the error and return a formatted error response
        print(f"Error in generate_with_api: {str(e)}")
        return {
            'id': 'api_error',
            'object': 'error',
            'created': None,
            'model': model_name_request,
            'content': f"Error: {str(e)}"
        }

def main(custom_model_name: str, num_samples: int = None, control_settings: Dict[str, float] = {}):
    # Load your model and tokenizer using the function from steering_api_functions
    model, tokenizer = load_model()

    model_name = custom_model_name
    user_tag = "[INST]"
    assistant_tag = "[/INST]"
    experimental_prompt = 'Give a truthful answer. '

    # Load TruthfulQA dataset
    questions, answers, labels = load_tqa_sentences(user_tag, assistant_tag, experimental_prompt, num_samples)
    print ('loaded dataset in main')

    # Calculate random accuracy
    random_acc = np.mean([1/len(l) for l in labels])
    print(f"Random Accuracy: {random_acc:.4f}")


    # Evaluate without control
    results_no_control = evaluate_tqa(
        generate_completion_response,
        model,
        tokenizer,
        questions,
        answers,
        labels,
        custom_model_name,
        control_settings={},
        generation_settings={"max_new_tokens": 256}
    )
    print(f"Standard Zero-Shot Accuracy: {results_no_control['accuracy']:.4f}")

    # Evaluate with truthful prompt (you can modify control_settings as needed)
    results_with_control = evaluate_tqa(
        generate_completion_response,
        model,
        tokenizer,
        questions,
        answers,
        labels,
        custom_model_name,
        control_settings=control_settings,  # Adjust this based on your control dimensions
        generation_settings={"max_new_tokens": 256}
    )
    print(f"Zero-Shot Accuracy with Truthful Control: {results_with_control['accuracy']:.4f}")

    # Save results to a JSON file
    results = {
        "random_accuracy": random_acc,
        "standard_zero_shot": results_no_control,
        "truthful_control": results_with_control
    }
    with open("tqa_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)


def test_generate_completion():
    model_name_request = "honest_1-b268"
    prompt = "Test prompt."
    control_settings = {"honesty": 1.0}
    generation_settings = {"max_new_tokens": 50}
    model, tokenizer = load_model()  # Ensure this function works and loads your model
    
    response = generate_completion_response(
        model_name_request=model_name_request,
        prompt=prompt,
        control_settings=control_settings,
        generation_settings=generation_settings,
        model=model,
        tokenizer=tokenizer
    )

    print("Generated response:", response)

def load_config():
    """
    Load configuration from eval_config.json file.
    Returns a list of configurations or default values if file is not found or cannot be parsed.
    """
    default_config = [{
        'custom_model_name': 'honest_1-b268',
        'control_settings': {"honesty": 1.0},
        'num_samples': None
    }]
    
    try:
        with open('eval_config.json', 'r') as config_file:
            configs = json.load(config_file)
        if not isinstance(configs, list):
            configs = [configs]  # Convert to list if it's a single dict
    except FileNotFoundError:
        print("eval_config.json not found. Using default values.")
        configs = default_config
    except json.JSONDecodeError:
        print("Error decoding eval_config.json. Using default values.")
        configs = default_config
    
    return configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with configuration from eval_config.json")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (overrides config file)")
    args = parser.parse_args()

    configs = load_config()

    for config in configs:
        custom_model_name = config.get('custom_model_name')
        control_settings = config.get('control_settings')
        num_samples = config.get('num_samples')

        # Command line argument overrides config file
        if args.num_samples is not None:
            num_samples = args.num_samples

        print(f"\nRunning evaluation with model: {custom_model_name}")
        print(f"Control settings: {control_settings}")
        print(f"Number of samples: {num_samples}")
        test_generate_completion()
        main(custom_model_name, num_samples=num_samples, control_settings=control_settings)
