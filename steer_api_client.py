import requests
import json
from dotenv import load_dotenv
import os
load_dotenv()
REMOTE_URL = os.getenv('REMOTE_URL')

# Function to create a steerable model
def create_steerable_model(model_label, control_dimensions, suffix_list=None):
    """Create a new steerable model and return its ID."""
    print (f"Creating steerable model with label: {model_label} and control dimensions: {control_dimensions}")
    payload = {
        "model_label": model_label,
        "control_dimensions": control_dimensions,
        "suffix_list": suffix_list or []
    }
    response = requests.post(f"{REMOTE_URL}/steerable-model", json=payload)
    if response.status_code == 201:
        return response.json()['id']
    else:
        raise Exception(f"Failed to create model: {response.text}")

# Function to list all steerable models
def list_steerable_models(limit=10, offset=0):
    """List all steerable models."""
    params = {'limit': limit, 'offset': offset}
    response = requests.get(f"{REMOTE_URL}/steerable-model", params=params)
    if response.status_code == 200:
        return response.json()['data']
    else:
        raise Exception(f"Failed to list models: {response.text}")

# Function to get details of a specific steerable model
def get_steerable_model(model_id):
    """Get details of a specific steerable model."""
    response = requests.get(f"{REMOTE_URL}/steerable-model/{model_id}")
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get model: {response.text}")

# Function to delete a specific steerable model
def delete_steerable_model(model_id):
    """Delete a specific steerable model."""
    response = requests.delete(f"{REMOTE_URL}/steerable-model/{model_id}")
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to delete model: {response.text}")

# Function to generate a completion using a specific model
def generate_completion(model_id, prompt, control_settings=None, settings=None):
    """Generate a completion using the specified model and prompt."""
    payload = {
        "model": model_id,
        "prompt": prompt,
        "control_settings": control_settings or {},
        "settings": settings or {"max_new_tokens": 186}
    }
    response = requests.post(f"{REMOTE_URL}/completions", json=payload)
    if response.status_code == 200:
        return response.json()['content']
    else:
        raise Exception(f"Failed to generate completion: {response.text}")

def health_check():
    """Check the health of the API by hitting the root endpoint."""
    try:
        response = requests.get(f"{REMOTE_URL}/")
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Health check failed with status code {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Health check request failed: {e}")
    