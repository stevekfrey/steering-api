import requests
import json
import time

# Base URL of the API
BASE_URL = 'http://localhost:5000'

# Test data for creating a steerable model
create_model_data = {
    "model_label": "testmodel",
    "control_dimensions": {
        "politeness": [
            ["polite", "courteous", "respectful"],
            ["rude", "impolite", "disrespectful"]
        ],
        "rude": [
            ["rude", "impolite", "disrespectful"],
            ["polite", "courteous", "respectful"]
        ]
    },
    "suffix_list": ["How are you?", "Thank you.", "Can I help you?", "What do you think?"]
}

# List of test prompts
test_prompts = [
    "Can you help me with my homework?",
    "I disagree with your opinion.",
    "What's the best way to learn a new language?",
    "Tell me a joke.",
    "Explain quantum computing in simple terms.",
    "Why is the sky blue?",
    "What's your opinion on artificial intelligence?",
    "How do I make friends as an adult?",
    "Describe the taste of an orange to someone who's never had one.",
    "What are the ethical implications of genetic engineering?"
]

def test_create_steerable_model():
    print("Testing: Create Steerable Model")
    response = requests.post(
        f"{BASE_URL}/steerable-models",
        headers={"Content-Type": "application/json"},
        data=json.dumps(create_model_data)
    )
    if response.status_code == 201:
        data = response.json()
        print("Steerable Model Created:", data)
        return data['id']
    else:
        print("Failed to create steerable model:", response.text)
        return None

def test_generate_completion(model_id, prompt, politeness, rudeness):
    print(f"\nTesting: Generate Completion for '{prompt}'")
    print(f"Politeness: {politeness}, Rudeness: {rudeness}")
    generate_completion_data = {
        "model": model_id,
        "prompt": prompt,
        "control_settings": {
            "politeness": politeness,
            "rude": rudeness
        },
        "settings": {}
    }
    response = requests.post(
        f"{BASE_URL}/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(generate_completion_data)
    )
    if response.status_code == 200:
        data = response.json()
        print("Generated Completion:")
        print(data['choices'][0]['text'])
    else:
        print("Failed to generate completion:", response.text)

def test_delete_steerable_model(model_id):
    print("\nTesting: Delete Steerable Model")
    response = requests.delete(f"{BASE_URL}/steerable-models/{model_id}")
    if response.status_code == 200:
        data = response.json()
        print("Deleted Steerable Model:", data)
    else:
        print("Failed to delete steerable model:", response.text)

if __name__ == "__main__":
    model_id = test_create_steerable_model()
    if model_id:
        for prompt in test_prompts:
            # Test with high rudeness, low politeness
            test_generate_completion(model_id, prompt, -2.0, 2.0)
            time.sleep(1)  # Add a small delay between requests
            
            # Test with high politeness, low rudeness
            test_generate_completion(model_id, prompt, 2.0, -2.0)
            time.sleep(1)  # Add a small delay between requests
        
        test_delete_steerable_model(model_id)
    else:
        print("Skipping further tests due to failure in creating steerable model.")
