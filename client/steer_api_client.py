import requests
import json
import os
from dotenv import load_dotenv
import time
import logging
import sys
import os
import typing
import backoff
import requests.exceptions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Common headers with Authorization
API_AUTH_TOKEN = os.getenv('API_AUTH_TOKEN')
if not API_AUTH_TOKEN:
    raise ValueError("API_AUTH_TOKEN is not set in the environment variables")
HEADERS = {
    'Authorization': f'Bearer {API_AUTH_TOKEN}'
}

REMOTE_URL = os.getenv('REMOTE_URL')
if not REMOTE_URL:
    raise ValueError("REMOTE_URL is not set in the environment variables")
logger.info(f"using REMOTE_URL: {REMOTE_URL}")

# Add these helper functions
def validate_response(response: requests.Response) -> None:
    """Validates the response from the API."""
    try:
        response.raise_for_status()
        if response.headers.get('Content-Type') != 'application/json':
            raise ValueError("Invalid Content-Type in response")
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, ValueError),
    max_tries=3
)
def make_api_request(
    method: str,
    url: str,
    headers: typing.Optional[typing.Dict[str, str]] = None,
    json: typing.Optional[typing.Dict[str, typing.Any]] = None,
    timeout: int = 30
) -> requests.Response:
    """Makes a robust API request with retries and validation."""
    headers = headers or {}
    headers.update(HEADERS)  # Add the authorization headers
    headers['Content-Type'] = 'application/json'
    
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=json,
            timeout=timeout
        )
        validate_response(response)
        return response
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        raise

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
    response = make_api_request("POST", url, json=payload)
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
    url = f"{REMOTE_URL}/steerable-model"
    params = {'limit': limit, 'offset': offset}
    response = make_api_request("GET", f"{url}?limit={limit}&offset={offset}")
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
    url = f"{REMOTE_URL}/completions"
    payload = {
        "model": model_id,
        "prompt": prompt,
        "control_settings": control_settings if control_settings is not None else {},
        "settings": settings if settings is not None else {"max_new_tokens": 186}
    }
    logger.info(f"Sending POST request to {url} with payload: {payload}")
    response = make_api_request("POST", url, json=payload)
    response.raise_for_status()
    return response.json()

def health_check():
    """Check the health of the API."""
    try:
        response = make_api_request("GET", f"{REMOTE_URL}/health", timeout=7)
        return response.json()
    except requests.exceptions.Timeout:
        raise Exception("The server is currently offline for maintenance.")
    except Exception as e:
        raise Exception(f"Health check failed: {str(e)}")
    

def get_model_status(model_id):
    url = f"{REMOTE_URL}/steerable-model/{model_id}/status"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def wait_for_model_ready(model_id, timeout=300, poll_interval=15):
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
