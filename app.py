from flask import Flask, request, jsonify
import logging
import traceback
import json_log_formatter
import os
import json
import time
from threading import Thread, Lock
import uuid
import numpy as np
import random

# Import functions from steering_api_functions.py
from steering_api_functions import (
    load_model,
    generate_completion_response,
    save_steerable_model,
    get_steerable_model,
    delete_steerable_model,
    create_dataset_and_train_vector,
    load_prompt_list
)
from steer_templates import (
    DEFAULT_TEMPLATE,
    BASE_MODEL_NAME,
    prompt_filepaths,
    DEFAULT_PROMPT_LIST
)

##############################################
# Flask setup
##############################################

app = Flask(__name__)

# Set up structured JSON logging
formatter = json_log_formatter.JSONFormatter()
json_handler = logging.StreamHandler()
json_handler.setFormatter(formatter)

app.logger.addHandler(json_handler)
app.logger.setLevel(logging.INFO)

# Add these constants near the top of the file
MODELS_DIR = 'steerable_models'
MODELS_FILE = os.path.join(MODELS_DIR, 'steerable_models.json')

################################################
# Load model
################################################

# Load the model at startup and store in app config
app.config['MODEL'], app.config['TOKENIZER'] = load_model()
app.config['MODEL_NAME'] = BASE_MODEL_NAME

# Load the default prompt type
PROMPT_TYPE = os.environ.get('PROMPT_TYPE', 'facts')
if PROMPT_TYPE not in prompt_filepaths:
    PROMPT_LIST = DEFAULT_PROMPT_LIST
else: 
    # Load the appropriate prompt list
    PROMPT_LIST = load_prompt_list(prompt_filepaths[PROMPT_TYPE])

app.logger.info(f"Loaded prompt list: {PROMPT_TYPE}\n{PROMPT_LIST}\n\n")

################################################
# In-memory model status dictionary
################################################

# Dictionary to store model statuses and metadata
STEERABLE_MODELS = {}

# Lock for thread-safe operations on STEERABLE_MODELS
steerable_models_lock = Lock()

'''
def save_models_to_json():
    os.makedirs(MODELS_DIR, exist_ok=True)
    serializable_models = {}
    for model_id, model_data in STEERABLE_MODELS.items():
        serializable_model = model_data.copy()
        if 'control_vectors' in serializable_model:
            serializable_model['control_vectors'] = {
                trait: vector.tolist() if isinstance(vector, np.ndarray) else vector
                for trait, vector in serializable_model['control_vectors'].items()
            }
        serializable_models[model_id] = serializable_model
    
    with open(MODELS_FILE, 'w') as f:
        json.dump(serializable_models, f)

def load_models_from_json():
    if os.path.exists(MODELS_FILE):
        with open(MODELS_FILE, 'r') as f:
            loaded_models = json.load(f)
        for model_id, model_data in loaded_models.items():
            if 'control_vectors' in model_data:
                model_data['control_vectors'] = {
                    trait: np.array(vector) if isinstance(vector, list) else vector
                    for trait, vector in model_data['control_vectors'].items()
                }
        return loaded_models
    return {}
'''

################################################
# Background Model Training Function
################################################

def train_steerable_model(model_id, model_label, control_dimensions, prompt_list):
    try:
        app.logger.info(f"Starting training for model {model_id}")
        
        # Update the model status to 'training'
        with steerable_models_lock:
            STEERABLE_MODELS[model_id]['status'] = 'training'

        # Perform the actual model training
        control_vectors = {}
        for trait, examples in control_dimensions.items():
            positive_examples = examples.get('positive_examples', [])
            negative_examples = examples.get('negative_examples', [])
            control_vectors[trait] = create_dataset_and_train_vector(
                positive_examples=positive_examples,
                negative_examples=negative_examples,
                prompt_list=prompt_list,
                template=DEFAULT_TEMPLATE,
                model=app.config['MODEL'],
                tokenizer=app.config['TOKENIZER']
            )

        # Update the model data with control vectors
        with steerable_models_lock:
            STEERABLE_MODELS[model_id]['control_vectors'] = control_vectors
            STEERABLE_MODELS[model_id]['status'] = 'ready'

        app.logger.info(f"Model {model_id} training completed.\n   Steering vectors created: {control_vectors.keys()}")

    except Exception as e:
        # Update status to 'failed' in case of any exception
        with steerable_models_lock:
            STEERABLE_MODELS[model_id]['status'] = 'failed'
        app.logger.error(f"Error training model {model_id}", extra={'error': str(e), 'traceback': traceback.format_exc()})

################################################
# Main Endpoints
################################################

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "success", "message": "Hello from Steer API"}), 200

# Endpoint to create a steerable model
@app.route('/steerable-model', methods=['POST'])
def create_steerable_model_endpoint():
    try:
        data = request.get_json()
        model_label = data.get('model_label')
        control_dimensions = data.get('control_dimensions')
        prompt_list = data.get('prompt_list', PROMPT_LIST)

        if not model_label or not control_dimensions:
            app.logger.warning('Invalid request: missing model_label or control_dimensions')
            return jsonify({'error': 'model_label and control_dimensions are required'}), 400

        # Generate a unique model_id
        unique_id = f"{random.randint(10, 99)}"  # Generate two random digits
        model_id = f"{model_label}-{unique_id}"

        # Create the initial model data
        model_response_data = {
            'id': model_id,
            'object': 'steerable_model',
            'created_at': int(time.time()),
            'model': app.config['MODEL_NAME'],
            'model_label': model_label,
            'control_dimensions': control_dimensions,
            'status': 'pending'
        }

        # Save the initial model data
        with steerable_models_lock:
            STEERABLE_MODELS[model_id] = model_response_data

        # Start the background thread for model training
        training_thread = Thread(
            target=train_steerable_model,
            args=(model_id, model_label, control_dimensions, prompt_list)
        )
        training_thread.start()

        # Return the initial model data to the user
        return jsonify(model_response_data), 202  # 202 Accepted

    except Exception as e:
        app.logger.error('Error in create_steerable_model', extra={'error': str(e), 'traceback': traceback.format_exc()})
        return jsonify({'error': 'An internal error occurred', 'details': str(e)}), 500

# Endpoint to list steerable models
@app.route('/steerable-model', methods=['GET'])
def list_models():
    limit = request.args.get('limit', default=10, type=int)
    offset = request.args.get('offset', default=0, type=int)

    with steerable_models_lock:
        models_list = list(STEERABLE_MODELS.values())[offset:offset+limit]

    # Prepare the response data without control vectors but including control_dimensions
    data = [{
        'id': model['id'],
        'object': model['object'],
        'created_at': model['created_at'],
        'model': model['model'],
        'control_dimensions': model.get('control_dimensions', {}),
        'status': model.get('status', 'unknown')
    } for model in models_list]

    app.logger.info('Steerable models listed', extra={'limit': limit, 'offset': offset})
    return jsonify({'data': data}), 200

# Endpoint to retrieve a specific steerable model
@app.route('/steerable-model/<model_id>', methods=['GET'])
def get_model(model_id):
    with steerable_models_lock:
        model = STEERABLE_MODELS.get(model_id)
    if not model:
        app.logger.warning('Steerable model not found', extra={'model_id': model_id})
        return jsonify({'error': 'Steerable model not found'}), 404

    # Return the model details without control vectors but including control_dimensions and status
    response = {
        'id': model['id'],
        'object': model['object'],
        'created_at': model['created_at'],
        'model': model['model'],
        'control_dimensions': model.get('control_dimensions', {}),
        'status': model.get('status', 'unknown')
    }
    app.logger.info('Steerable model retrieved', extra={'model_id': model_id})
    return jsonify(response), 200

# Endpoint to delete a steerable model
@app.route('/steerable-model/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    with steerable_models_lock:
        if model_id in STEERABLE_MODELS:
            # Perform any necessary cleanup (e.g., delete model files)
            delete_steerable_model(model_id)
            del STEERABLE_MODELS[model_id]
            app.logger.info('Steerable model deleted', extra={'model_id': model_id})
            return jsonify({
                'id': model_id,
                'object': 'steerable_model',
                'deleted': True
            }), 200
        else:
            app.logger.warning('Steerable model not found for deletion', extra={'model_id': model_id})
            return jsonify({'error': 'Steerable model not found'}), 404

# Endpoint to check the status of a model
@app.route('/steerable-model/<model_id>/status', methods=['GET'])
def check_model_status(model_id):
    with steerable_models_lock:
        model = STEERABLE_MODELS.get(model_id)
    if not model:
        app.logger.warning('Steerable model not found', extra={'model_id': model_id})
        return jsonify({'error': 'Steerable model not found'}), 404

    status = {'id': model_id, 'status': model.get('status', 'unknown')}
    return jsonify(status), 200

@app.route('/completions', methods=['POST'])
def generate_completion():
    try:
        data = request.get_json()
        model_name_request = data.get('model')
        prompt = data.get('prompt')
        control_settings = data.get('control_settings', {})
        generation_settings = data.get('settings', {})

        if not model_name_request or not prompt:
            app.logger.warning('Invalid request: missing model or prompt')
            return jsonify({'error': 'Model and prompt are required'}), 400

        app.logger.info(f"Received request to create generation with model: {model_name_request}\nand prompt: {prompt}")
        
        # Check if the specified model is ready
        with steerable_models_lock:
            model = STEERABLE_MODELS.get(model_name_request)
            if model:
                if model.get('status') != 'ready':
                    app.logger.warning('Model not ready', extra={'model_id': model_name_request})
                    return jsonify({'error': f'Model {model_name_request} is not ready'}), 400
            else:
                app.logger.warning('Model not found', extra={'model_id': model_name_request})
                return jsonify({'error': f'Model {model_name_request} not found'}), 404

        # Log input data
        app.logger.info('*** Completion requested', extra={'model': model_name_request, 'prompt': prompt})

        # Generate the completion response, passing control_vectors
        response = generate_completion_response(
            model_name_request,
            prompt,
            control_settings,
            generation_settings,
            app.config['MODEL'],
            app.config['TOKENIZER'],
            control_vectors=model.get('control_vectors')
        )

        return jsonify(response), 200

    except Exception as e:
        app.logger.error('Error in generate_completion', extra={'error': str(e), 'traceback': traceback.format_exc()})
        return jsonify({'error': 'An internal error occurred', 'details': str(e)}), 500

# Global error handler
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error('Unhandled exception', extra={'error': str(e), 'traceback': traceback.format_exc()})
    response = {
        'error': 'An internal error occurred',
        'details': str(e)
    }
    return jsonify(response), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
