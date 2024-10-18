import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.steer_api_client import (
    create_steerable_model,
    list_steerable_models,
    get_steerable_model,
    generate_completion,
    wait_for_model_ready
)

# Sample data for creating a steerable model
prompt_list = [
    "You discover a hidden compartment in your car.",
    "You find a book in your library you don't remember buying.",
    "You stumble upon a shortcut on your regular walking route."
]

create_model_data = {
    "prompt_list": prompt_list,
    "model_label": "testmodel",
    "control_dimensions": {
        "materialistic": {
            "positive_examples": ["materialistic", "consumerist", "acquisitive"],
            "negative_examples": ["spiritual", "altruistic", "ascetic"]
        },
        "optimistic": {
            "positive_examples": ["optimistic", "happy go lucky", "cheerful"],
            "negative_examples": ["pessimistic", "gloomy", "negative"]
        }
    }
}

test_prompts = [
    "You successfully complete a dare but realize how dangerous it was.",
    "Your pet goes missing but returns carrying something mysterious in its mouth.",
    "You find a secluded beach but the tides start to rise quickly.",
    "You're invited to speak at an event in front of a large audience.",
    "You receive a scholarship but it requires you to move abroad."
]

def main():
    # Create a new steerable model
    model_info = create_steerable_model(
        create_model_data["model_label"],
        create_model_data["control_dimensions"],
        create_model_data["prompt_list"]
    )
    model_id = model_info['id']
    print(f"Created steerable model with ID: {model_id}")

    # Wait for the model to be ready
    wait_for_model_ready(model_id)

    # List all steerable models
    models = list_steerable_models()
    print(f"Available models: {json.dumps(models, indent=2)}")

    # Get details of the created model
    model_details = get_steerable_model(model_id)
    print(f"Model details: {json.dumps(model_details, indent=2)}")

    # Test prompts with different control settings
    control_settings = [
        {},  # No control
        {"materialistic": 2, "optimistic": 2},  # Increase both dimensions
        {"materialistic": -2, "optimistic": -2}  # Decrease both dimensions
    ]

    for prompt in test_prompts:
        print(f"\nTesting prompt: {prompt}")
        for setting in control_settings:
            print(f"\nControl settings: {setting}")
            response = generate_completion(model_id, prompt, setting)
            print(f"Generated response: {response['content']}")

if __name__ == "__main__":
    main()
