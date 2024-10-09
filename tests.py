import unittest
import requests
import json
import time

BASE_URL = 'http://localhost:5000'

user_tag = "<|start_header_id|>user<|end_header_id|>You: "
asst_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>Assistant: "

DEFAULT_TEMPLATE = "I am a {persona} person."

# Control strength variables
TEST_CONTROL_INCREASE = 2
TEST_CONTROL_DECREASE = -2

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

# Test prompts to be used for generation
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

class TestAppEndpoints(unittest.TestCase):
    # Declare class variables
    steerable_model_id = None

    @classmethod
    def setUpClass(cls):
        """Create a steerable model once for all tests."""
        url = f"{BASE_URL}/steerable-models"
        response = requests.post(url, json=create_model_data)
        if response.status_code == 201:
            data = response.json()
            cls.steerable_model_id = data['id']
            print(f"Created steerable model with ID: {cls.steerable_model_id}")
            # Wait until the model is ready
            for _ in range(30):  # Wait up to 30 seconds
                model_url = f"{BASE_URL}/steerable-models/{cls.steerable_model_id}"
                model_response = requests.get(model_url)
                if model_response.status_code == 200:
                    model_data = model_response.json()
                    if model_data.get('status') == 'ready':
                        print("Model is ready for use.")
                        break
                time.sleep(1)
            else:
                raise Exception("Model did not become ready in time.")
        else:
            raise Exception(f"Failed to create steerable model: {response.status_code} - {response.text}")

    @classmethod
    def tearDownClass(cls):
        """Delete the steerable model after all tests."""
        url = f"{BASE_URL}/steerable-models/{cls.steerable_model_id}"
        response = requests.delete(url)
        if response.status_code == 200:
            data = response.json()
            if data.get('deleted'):
                print(f"Deleted steerable model with ID: {cls.steerable_model_id}")
            else:
                print("Failed to delete steerable model.")
        else:
            print(f"Failed to delete steerable model: {response.status_code} - {response.text}")

    def test_get_steerable_model(self):
        """Test retrieving a specific steerable model."""
        model_id = self.__class__.steerable_model_id
        print(f"Testing get_steerable_model with ID: {model_id}")
        url = f"{BASE_URL}/steerable-models/{model_id}"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200, msg=f"Status code: {response.status_code}, Response: {response.text}")
        data = response.json()
        self.assertEqual(data['id'], model_id)
        print(f"Retrieved steerable model with ID: {model_id}")

    def test_list_steerable_models(self):
        """Test listing steerable models."""
        url = f"{BASE_URL}/steerable-models"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200, msg=f"Status code: {response.status_code}, Response: {response.text}")
        data = response.json()
        self.assertIn('data', data)
        print(f"Listed steerable models: {[model['id'] for model in data['data']]}")

    def generate_completion_with_prompt(self, model_id, prompt, control_settings, settings=None):
        """Helper function to generate completion."""
        url = f"{BASE_URL}/completions"
        payload = {
            "model": model_id,
            "prompt": prompt,
            "control_settings": control_settings,
            "settings": settings or {}
        }
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"Error generating completion: {response.status_code} - {response.text}")
        return response

    def test_generation_with_control_strength(self):
        """Test generating completions with different control strengths."""
        model_id = self.__class__.steerable_model_id
        actions = [
            ("Prompt only", 0),
            ("Prompt + control", TEST_CONTROL_INCREASE),
            ("Prompt - control", TEST_CONTROL_DECREASE)
        ]

        for prompt in test_prompts:
            for action_name, control_strength in actions:
                print(f"\n---- {action_name}, control strength: {control_strength} ----\n")
                # Adjust control settings based on action
                if control_strength == 0:
                    control_settings = {}
                else:
                    control_settings = {
                        "materialistic": control_strength,
                        "optimistic": control_strength
                    }
                print(f"Testing generation with model ID: {model_id}")
                response = self.generate_completion_with_prompt(
                    model_id=model_id,
                    prompt=prompt,
                    control_settings=control_settings
                )
                self.assertEqual(response.status_code, 200, msg=f"Status code: {response.status_code}, Response: {response.text}")
                data = response.json()
                self.assertIn('content', data)
                # Check if content is empty
                if not data['content']:
                    print("Warning: Generated content is empty.")
                # Format and print the response
                content = data['content'].replace(user_tag, "\nUser: ").replace(asst_tag, "\nAssistant: ").strip()
                print("Control settings:", control_settings)
                print(f"Prompt: {prompt}\n")
                print(f"Generated Response:\n{content}\n")
                print('-' * 80)

    def test_delete_steerable_model(self):
        """Test deleting a steerable model."""
        model_id = self.__class__.steerable_model_id
        url = f"{BASE_URL}/steerable-models/{model_id}"
        response = requests.delete(url)
        self.assertEqual(response.status_code, 200, msg=f"Status code: {response.status_code}, Response: {response.text}")
        data = response.json()
        self.assertTrue(data.get('deleted'))
        print(f"Deleted steerable model with ID: {model_id}")

if __name__ == '__main__':
    unittest.main()