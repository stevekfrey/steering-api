# Test suite that directly calls the Flask endpoints in app.py using the test client.
import time
import datetime
import uuid
import json

# Import the Flask app instance from app.py
from app import app

# Initialize the test client
client = app.test_client()

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
    data = create_model_data
    # Use the test client to send a POST request to the endpoint
    response = client.post('/steerable-models',
                           data=json.dumps(data),
                           content_type='application/json')
    result = response.get_json()
    if response.status_code == 201 and 'error' not in result:
        print("Steerable Model Created:", result)
        return result['id']
    else:
        print("Failed to create steerable model:", result.get('error', 'Unknown error'))
        return None

def test_generate_completion(model_id, prompt, politeness, rudeness):
    print(f"\nTesting: Generate Completion for '{prompt}'")
    print(f"Politeness: {politeness}, Rudeness: {rudeness}")
    data = {
        "model": model_id,
        "prompt": prompt,
        "control_settings": {
            "politeness": politeness,
            "rude": rudeness
        },
        "settings": {}
    }
    # Use the test client to send a POST request to the endpoint
    response = client.post('/completions',
                           data=json.dumps(data),
                           content_type='application/json')
    result = response.get_json()
    if response.status_code == 200 and 'error' not in result:
        print("Generated Completion:")
        print(result['choices'][0]['text'])
    else:
        print("Failed to generate completion:", result.get('error', 'Unknown error'))

def test_delete_steerable_model(model_id):
    print("\nTesting: Delete Steerable Model")
    # Use the test client to send a DELETE request to the endpoint
    response = client.delete(f'/steerable-models/{model_id}')
    result = response.get_json()
    if response.status_code == 200 and 'error' not in result:
        print("Deleted Steerable Model:", result)
    else:
        print("Failed to delete steerable model:", result.get('error', 'Unknown error'))

if __name__ == "__main__":
    model_id = test_create_steerable_model()
    if model_id:
        for prompt in test_prompts:
            # Test with high rudeness, low politeness
            test_generate_completion(model_id, prompt, -2.0, 2.0)
            time.sleep(1)  # Add a small delay between calls

            # Test with high politeness, low rudeness
            test_generate_completion(model_id, prompt, 2.0, -2.0)
            time.sleep(1)  # Add a small delay between calls

        test_delete_steerable_model(model_id)
    else:
        print("Skipping further tests due to failure in creating steerable model.")