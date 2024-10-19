import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.steering_api_functions import (
    load_model,
    create_steerable_model_honesty,
    list_steerable_models,
    get_steerable_model,
    generate_completion_response_honesty,
    wait_for_model_ready
)

# Path to the facts_true_false.csv dataset
data_path = "path/to/facts_true_false.csv"

def main():
    # Load the base model and tokenizer
    model, tokenizer = load_model()

    # Create a new steerable model using the honesty dataset
    model_label = "honesty_model"
    model_info = create_steerable_model_honesty(
        model_label,
        data_path,
        model,
        tokenizer
    )
    model_id = model_info['id']
    print(f"Created steerable model with ID: {model_id}")

    # Wait for the model to be ready (if needed)
    wait_for_model_ready(model_id)

    # List all steerable models
    models = list_steerable_models()
    print(f"Available models: {json.dumps(models, indent=2)}")

    # Get details of the created model
    model_details = get_steerable_model(model_id)
    print(f"Model details: {json.dumps(model_details, indent=2)}")

    # Define test prompts
    test_prompts = [
        "Tell me something about the moon.",
        "What can you say about the Earth?",
        "Explain the theory of evolution.",
    ]

    # Define control settings for honesty
    control_settings_list = [
        {},  # No control
        {"honesty": 2},  # Increase honesty
        {"honesty": -2},  # Decrease honesty (promote dishonesty)
    ]

    # Test the model with different control settings
    for prompt in test_prompts:
        print(f"\nTesting prompt: {prompt}")
        for control_settings in control_settings_list:
            print(f"\nControl settings: {control_settings}")
            response = generate_completion_response_honesty(
                model_id,
                prompt,
                control_settings,
                generation_settings={},
                model=model,
                tokenizer=tokenizer
            )
            print(f"Generated response: {response['content']}")

if __name__ == "__main__":
    main()