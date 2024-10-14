from flask import Flask, request, jsonify
import logging
import traceback
import json_log_formatter

# Import functions from steering_api_functions.py
from steering_api_functions import (
    load_model,
    create_dataset_and_train_vector,
    generate_completion_response,
    save_steerable_model,
    get_steerable_model,
    list_steerable_models,
    delete_steerable_model,
    create_steerable_model_function, 
    load_prompt_list,
    create_steerable_model_async,
    get_model_status
)
from steer_templates import (
    DEFAULT_TEMPLATE,
    BASE_MODEL_NAME,
    prompt_filepaths,  # Import prompt_filepaths
    DEFAULT_PROMPT_LIST  # Import DEFAULT_PROMPT_LIST
)

##############################################
# Flask setup
################################################

app = Flask(__name__)

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

# Load the appropriate suffix list
DEFAULT_SUFFIX_LIST = load_prompt_list(prompt_filepaths[DEFAULT_PROMPT_LIST])

################################################
# Main Endpoints 
################################################

@app.route('/', methods=['GET'])
def index():
    return "Hello from Steer-API!"

# Endpoint to create a steerable model
@app.route('/steerable-model', methods=['POST'])
def create_steerable_model():
    try:
        data = request.get_json()
        model_label = data.get('model_label')
        control_dimensions = data.get('control_dimensions')
        suffix_list = data.get('suffix_list', DEFAULT_SUFFIX_LIST)

        response = create_steerable_model_async(
            model_label=model_label,
            control_dimensions=control_dimensions,
            suffix_list=suffix_list,
            model=app.config['MODEL'],
            tokenizer=app.config['TOKENIZER']
        )

        return jsonify(response), 202  # 202 Accepted

    except Exception as e:
        app.logger.error('Error in create_steerable_model', extra={'error': str(e), 'traceback': traceback.format_exc()})
        return jsonify({'error': 'An internal error occurred', 'details': str(e)}), 500

# Endpoint to list steerable models
@app.route('/steerable-model', methods=['GET'])
def list_models():
    limit = request.args.get('limit', default=10, type=int)
    offset = request.args.get('offset', default=0, type=int)

    models_list = list_steerable_models(limit, offset)

    # Prepare the response data without control vectors but including control_dimensions
    data = [{
        'id': model['id'],
        'object': model['object'],
        'created_at': model['created_at'],
        'model': model['model'],
        'control_dimensions': model.get('control_dimensions', {})
    } for model in models_list]

    app.logger.info('Steerable models listed', extra={'limit': limit, 'offset': offset})
    return jsonify({'data': data}), 200

# Endpoint to retrieve a specific steerable model
@app.route('/steerable-model/<model_id>', methods=['GET'])
def get_model(model_id):
    model = get_steerable_model(model_id)
    if not model:
        app.logger.warning('Steerable model not found', extra={'model_id': model_id})
        return jsonify({'error': 'Steerable model not found'}), 404

    # Return the model details without control vectors but including control_dimensions
    response = {
        'id': model['id'],
        'object': model['object'],
        'created_at': model['created_at'],
        'model': model['model'],
        'control_dimensions': model.get('control_dimensions', {})
    }
    app.logger.info('Steerable model retrieved', extra={'model_id': model_id})
    return jsonify(response), 200

# Endpoint to delete a steerable model
@app.route('/steerable-model/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    success = delete_steerable_model(model_id)
    if success:
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
    status = get_model_status(model_id)
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
