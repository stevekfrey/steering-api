import requests
import json
import os
from dotenv import load_dotenv
import time
import logging

# Load environment variables
load_dotenv()
REMOTE_URL = os.getenv('REMOTE_URL')
API_AUTH_TOKEN = os.getenv('API_AUTH_TOKEN')

if not REMOTE_URL:
    raise ValueError("REMOTE_URL is not set in the environment variables")

if not API_AUTH_TOKEN:
    raise ValueError("API_AUTH_TOKEN is not set in the environment variables")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common headers with Authorization
HEADERS = {
    'Authorization': f'Bearer {API_AUTH_TOKEN}'
}

# Function to create a steerable model
def create_steerable_model(model_label, control_dimensions, prompt_list=None):
    """Create a new steerable model."""
    url = f"{REMOTE_URL}/steerable-model"
    payload = {
        "model_label": model_label,
        "control_dimensions": control_dimensions,
        "prompt_list": prompt_list
    }
    logger.info(f"Sending POST request to {url} with payload: {payload}")
    response = requests.post(url, json=payload, headers=HEADERS)
    if response.status_code == 202:
        model_info = response.json()
        logger.info(f"Received response: {model_info}")
        return model_info
    else:
        logger.error(f"Request failed with status code {response.status_code}: {response.text}")
        response.raise_for_status()

# Function to list all steerable models
def list_steerable_models(limit=20, offset=0):
    """List all steerable models."""
    params = {'limit': limit, 'offset': offset}
    response = requests.get(f"{REMOTE_URL}/steerable-model", params=params, headers=HEADERS)
    response.raise_for_status()
    return response.json()['data']

# Function to get details of a specific steerable model
def get_steerable_model(model_id):
    """Get details of a specific steerable model."""
    response = requests.get(f"{REMOTE_URL}/steerable-model/{model_id}", headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get model: {response.text}")

# Function to delete a specific steerable model
def delete_steerable_model(model_id):
    """Delete a specific steerable model."""
    response = requests.delete(f"{REMOTE_URL}/steerable-model/{model_id}", headers=HEADERS)
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
        "control_settings": control_settings if control_settings is not None else {},
        "settings": settings if settings is not None else {"max_new_tokens": 186}
    }
    response = requests.post(f"{REMOTE_URL}/completions", json=payload, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def health_check():
    """Check the health of the API by hitting the root endpoint."""
    try:
        response = requests.get(f"{REMOTE_URL}/", headers=HEADERS)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Health check failed with status code {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Health check request failed: {e}")
def get_model_status(model_id):
    url = f"{REMOTE_URL}/steerable-model/{model_id}/status"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def wait_for_model_ready(model_id, timeout=300, poll_interval=10):
    """Poll the model status until it's ready or timeout is reached."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        status_info = get_model_status(model_id)
        status = status_info.get('status')
        if status == 'ready':
            print(f"Model {model_id} is ready.")
            return True
        elif status == 'failed':
            raise Exception(f"Model {model_id} training failed.")
        else:
            print(f"Waiting for model {model_id} to be ready. Current status: {status}")
            time.sleep(poll_interval)
    raise TimeoutError(f"Timeout: Model {model_id} is not ready after {timeout} seconds.")
