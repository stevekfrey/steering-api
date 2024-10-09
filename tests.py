import requests
import json
import time

# Base URL of the API
BASE_URL = 'http://localhost:5000'

user_tag = "<|start_header_id|>user<|end_header_id|>You: "
asst_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>Assistant:"
DEFAULT_TEMPLATE = "I am a {persona} person."

# Test data for creating a steerable model
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
    },
        "suffix_list": [
        "", "That game", "I can see", "Hmm, this", "I can relate to", "Who is",
        "I understand the", "Ugh,", "What the hell was", "Hey, did anyone", "Although",
        "Thank you for choosing", "What are you", "Oh w", "How dare you open",
        "It was my pleasure", "I'm hon", "I appreciate that you", "Are you k",
        "Whoever left this", "It's always", "Ew,", "Hey, I l", "Hello? Is someone",
        "I understand that", "That poem", "Aww, poor", "Hey, it", "Alright, who",
        "I didn't", "Well, life", "The document", "Oh no, this", "I'm concerned",
        "Hello, this is", "This art", "Hmm, this drink", "Hi there!", "It seems",
        "Is", "Good", "I can't", "Ex", "Who are", "I can see that", "Wow,",
        "Today is a", "Hey friend", "Sometimes friends"
    ]
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

def test_generate_completion(model_id, prompt, control_settings):
    full_prompt = f"{user_tag}{prompt}{asst_tag}"
    
    generate_completion_data = {
        "model": model_id,
        "prompt": full_prompt,
        "control_settings": control_settings,
        "settings": {}
    }
    print(f"Test with Control Settings: {control_settings}")
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
        test_prompts = [
            "Can you help me with my homework?",
            "I disagree with your opinion.",
            "What's the best way to learn a new language?",
            "Tell me a joke.",
            "Explain quantum computing in simple terms."
        ]
        
        for prompt in test_prompts:
            # Test with high materialistic, cold, and selfish traits
            control_settings = {
                'materialistic': 2.0,
                'optimistic': 2.0,
            }
            test_generate_completion(model_id, prompt, control_settings)
            time.sleep(1)  # Add a small delay between requests
            
            # Test with low materialistic, cold, and selfish traits
            control_settings = {
                'materialistic': -2.0,
                'optimistic': -2.0,
            }
            test_generate_completion(model_id, prompt, control_settings)
            time.sleep(1)  # Add a small delay between requests
        
        test_delete_steerable_model(model_id)
    else:
        print("Skipping further tests due to failure in creating steerable model.")