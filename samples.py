import requests
import json

BASE_URL = 'http://localhost:5000'

# Sample data for creating a steerable model
create_model_data = {
    "model_label": "testmodel",
    "control_dimensions": {
        "materialistic": [
            ["materialistic", "consumerist", "acquisitive"],
            ["spiritual", "altruistic", "ascetic"]
        ],
        "optimistic": [
            ["optimistic", "happy go lucky", "cheerful"],
            ["pessimistic", "gloomy", "negative"]
        ]
    }
}

# Test prompts
test_prompts = [
    "Can you help me with my homework?",
    "I disagree with your opinion.",
    "What's the best way to learn a new language?",
    "Tell me a joke.",
    "Explain quantum computing in simple terms."
]

def create_steerable_model():
    """Create a new steerable model and return its ID."""
    response = requests.post(f"{BASE_URL}/steerable-models", json=create_model_data)
    if response.status_code == 201:
        return response.json()['id']
    else:
        raise Exception(f"Failed to create model: {response.text}")

def list_steerable_models():
    """List all steerable models."""
    response = requests.get(f"{BASE_URL}/steerable-models")
    if response.status_code == 200:
        return response.json()['data']
    else:
        raise Exception(f"Failed to list models: {response.text}")

def get_steerable_model(model_id):
    """Get details of a specific steerable model."""
    response = requests.get(f"{BASE_URL}/steerable-models/{model_id}")
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get model: {response.text}")

def generate_completion(model_id, prompt, control_settings=None):
    """Generate a completion using the specified model and prompt."""
    payload = {
        "model": model_id,
        "prompt": prompt,
        "control_settings": control_settings or {}
    }
    response = requests.post(f"{BASE_URL}/completions", json=payload)
    if response.status_code == 200:
        return response.json()['content']
    else:
        raise Exception(f"Failed to generate completion: {response.text}")

def main():
    # Create a new steerable model
    model_id = create_steerable_model()
    print(f"Created steerable model with ID: {model_id}")

    # List all steerable models
    models = list_steerable_models()
    print(f"Available models: {json.dumps(models, indent=2)}")

    # Get details of the created model
    model_details = get_steerable_model(model_id)
    print(f"Model details: {json.dumps(model_details, indent=2)}")

    # Test prompts with different control settings
    test_control_settings = [
        {},  # No control
        {"materialistic": 2, "optimistic": 2},  # Increase both dimensions
        {"materialistic": -2, "optimistic": -2}  # Decrease both dimensions
    ]

    for prompt in test_prompts:
        print(f"\nTesting prompt: {prompt}")
        for setting in test_control_settings:
            print(f"\nControl settings: {setting}")
            response = generate_completion(model_id, prompt, setting)
            print(f"Generated response: {response}")

if __name__ == "__main__":
    main()