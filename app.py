from flask import Flask, request, jsonify
import logging
import traceback
import json_log_formatter
from celery import Celery
import os
import json
import time

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
# Flask and Celery setup
##############################################

app = Flask(__name__)

# Celery configuration
app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Set up structured JSON logging
formatter = json_log_formatter.JSONFormatter()
json_handler = logging.StreamHandler()
json_handler.setFormatter(formatter)
app.logger.addHandler(json_handler)
app.logger.setLevel(logging.INFO)

################################################
# Load model
################################################

# Load the model at startup and store in app config
app.config['MODEL'], app.config['TOKENIZER'] = load_model()
app.config['MODEL_NAME'] = BASE_MODEL_NAME

# Load the appropriate prompt list
PROMPT_LIST = load_prompt_list(prompt_filepaths[DEFAULT_PROMPT_LIST])

################################################
# In-memory model status dictionary and JSON storage
################################################

# Dictionary to store model statuses and other metadata
STEERABLE_MODELS = {}

# Path to the JSON file for persisting model data
STEERABLE_MODELS_JSON_PATH = 'steerable_models.json'

# Load existing models from JSON file if it exists
if os.path.exists(STEERABLE_MODELS_JSON_PATH):
    with open(STEERABLE_MODELS_JSON_PATH, 'r') as f:
        STEERABLE_MODELS = json.load(f)
else:
    # Initialize an empty JSON file
    with open(STEERABLE_MODELS_JSON_PATH, 'w') as f:
        json.dump(STEERABLE_MODELS, f)

def save_models_to_json():
    """Utility function to save the STEERABLE_MODELS dict to JSON file."""
    with open(STEERABLE_MODELS_JSON_PATH, 'w') as f:
        json.dump(STEERABLE_MODELS, f)

################################################
# Celery task for model training
################################################

@celery.task(bind=True)
def train_steerable_model_task(self, model_id, model_label, control_dimensions, prompt_list):
    try:
        app.logger.info(f"Starting training for model {model_id}")

        # Update status to 'training' in the in-memory dict and JSON file
        STEERABLE_MODELS[model_id]['status'] = 'training'
        save_models_to_json()

        # Perform the actual model training
        create_dataset_and_train_vector(
            model_id=model_id,
            model_label=model_label,
            control_dimensions=control_dimensions,
            prompt_list=prompt_list,
            model=app.config['MODEL'],
            tokenizer=app.config['TOKENIZER']
        )

        # Update status to 'ready' after training completes
        STEERABLE_MODELS[model_id]['status'] = 'ready'
        save_models_to_json()

        app.logger.info(f"Model {model_id} training completed")

    except Exception as e:
        # Update status to 'failed' in case of any exception
        STEERABLE_MODELS[model_id]['status'] = 'failed'
        save_models_to_json()
        app.logger.error(f"Error training model {model_id}", extra={'error': str(e), 'traceback': traceback.format_exc()})
        raise e

################################################
# Main Endpoints
################################################

@app.route('/', methods=['GET'])
def index():
    return "Hello from Steer-API!"

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

        # Generate a unique model_id (you can use any method; here we use UUID)
        import uuid
        model_id = str(uuid.uuid4())

        # Save the initial model data with status 'pending' in the in-memory dict and JSON file
        model_data = {
            'id': model_id,
            'object': 'steerable_model',
            'created_at': int(time.time()),
            'model': app.config['MODEL_NAME'],
            'model_label': model_label,
            'control_dimensions': control_dimensions,
            'status': 'pending'
        }
        STEERABLE_MODELS[model_id] = model_data
        save_models_to_json()

        # Start the Celery task for model training
        task = train_steerable_model_task.delay(model_id, model_label, control_dimensions, prompt_list)

        # Return the initial model data to the user
        response = {
            'id': model_id,
            'object': 'steerable_model',
            'created_at': model_data['created_at'],
            'model': app.config['MODEL_NAME'],
            'model_label': model_label,
            'control_dimensions': control_dimensions,
            'status': 'pending'
        }

        return jsonify(response), 202  # 202 Accepted

    except Exception as e:
        app.logger.error('Error in create_steerable_model', extra={'error': str(e), 'traceback': traceback.format_exc()})
        return jsonify({'error': 'An internal error occurred', 'details': str(e)}), 500

# Endpoint to list steerable models
@app.route('/steerable-model', methods=['GET'])
def list_models():
    limit = request.args.get('limit', default=10, type=int)
    offset = request.args.get('offset', default=0, type=int)

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
    if model_id in STEERABLE_MODELS:
        delete_steerable_model(model_id)
        del STEERABLE_MODELS[model_id]
        save_models_to_json()

        app.logger.info('Steerable model deleted', extra={'model_id': model_id})
        return jsonify({
            'id': model_id,
            'object': 'steerable_model',
            'deleted': True
        }), 200
    else:
        app.logger.warning('Steerable model not found for deletion', extra={'model_id': model_id})
        return jsonify({'error': 'Steerable model not found'}), 404

@app.route('/steerable-model/<model_id>/status', methods=['GET'])
def check_model_status(model_id):
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

        # Check if the specified model is ready
        model = STEERABLE_MODELS.get(model_name_request)
        if model:
            if model.get('status') != 'ready':
                app.logger.warning('Model not ready', extra={'model_id': model_name_request})
                return jsonify({'error': f'Model {model_name_request} is not ready'}), 400
        else:
            app.logger.warning('Model not found', extra={'model_id': model_name_request})
            return jsonify({'error': f'Model {model_name_request} not found'}), 404

        # Log input data
        app.logger.info('Completion requested', extra={'model': model_name_request, 'prompt': prompt})

        # Generate the completion response
        response = generate_completion_response(
            model_name_request,
            prompt,
            control_settings,
            generation_settings,
            app.config['MODEL'],
            app.config['TOKENIZER']
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
    app.run(host='0.0.0.0', port=5000)
