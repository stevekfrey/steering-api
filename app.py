from flask import Flask, request, jsonify, current_app
import uuid
import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel, DatasetEntry
import warnings
import numpy as np
import traceback
import json_log_formatter
import logging
from steer_templates import DEFAULT_TEMPLATE, DEFAULT_SUFFIX_LIST, user_tag, asst_tag, BASE_MODEL_NAME

# Set up structured JSON logging
formatter = json_log_formatter.JSONFormatter()

json_handler = logging.StreamHandler()
json_handler.setFormatter(formatter)

################################################
# Flask setup
################################################
app = Flask(__name__)
app.logger.addHandler(json_handler)
app.logger.setLevel(logging.INFO)

################################################
# Load model 
################################################
warnings.filterwarnings('ignore')
torch.cuda.empty_cache()

def load_model():
    # Model loading code
    model_name = BASE_MODEL_NAME
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    base_model = AutoModelForCausalLM.from_pretrained("steer-api/local_models", device_map="auto", local_files_only=True)
    
    app.logger.info("Loaded base model from steer-api/local_models")
    
    # Wrap the model with ControlModel
    control_layers = list(range(-5, -18, -1))
    model = ControlModel(base_model, control_layers)
    
    return model, tokenizer

# Load the model at startup and store in app config
with app.app_context():
    app.config['MODEL'], app.config['TOKENIZER'] = load_model()
    app.config['MODEL_NAME'] = BASE_MODEL_NAME 

################################################
# Data Helpers 
################################################
# In-memory storage for steerable models and their control vectors
steerable_models_vector_storage = {}

# Create a contrastive dataset
def make_contrastive_dataset(
    positive_personas: list,
    negative_personas: list,
    suffix_list: list,
    template: str = DEFAULT_TEMPLATE
) -> list:
    # Access the tokenizer from app config
    tokenizer = current_app.config['TOKENIZER']

    dataset = []
    for suffix in suffix_list:
        tokens = tokenizer.tokenize(suffix)
        for i in range(1, len(tokens)):
            truncated_suffix = tokenizer.convert_tokens_to_string(tokens[:i])
            for positive_persona, negative_persona in zip(positive_personas, negative_personas):
                positive_template = template.format(persona=positive_persona)
                negative_template = template.format(persona=negative_persona)
                dataset.append(
                    DatasetEntry(
                        positive=f"{user_tag} {positive_template} {asst_tag} {truncated_suffix}",
                        negative=f"{user_tag} {negative_template} {asst_tag} {truncated_suffix}",
                    )
                )
    return dataset

# def print_chat(full_string, role="assistant"):
#     for element in full_string.split(f"<|start_header_id|>{role}<|end_header_id|>")[1:]:
#         print(element.strip("<|eot_id|>"))

def parse_assistant_response(full_string):
    # Split the string and get the last part (assistant's response)
    # parts = full_string.split("<|start_header_id|>assistant<|end_header_id|>")
    parts = full_string.split("Assistant:")
    if len(parts) > 1:
        response = parts[-1].strip()
        # Remove any trailing <|eot_id|> tag
        response = response.rstrip("<|eot_id|>").strip()
        return response
    return ""

# Mimics the zeros_like from torch, for ControlVector
def create_vector_with_zeros_like(control_vector):
    zero_directions = {k: torch.zeros_like(torch.tensor(v)) if isinstance(v, np.ndarray) else torch.zeros_like(v)
                       for k, v in control_vector.directions.items()}
    return ControlVector(model_type=control_vector.model_type, directions=zero_directions)


# Create dataset and train control vector
def create_dataset_and_train_vector(synonyms, antonyms, suffix_list, template=DEFAULT_TEMPLATE):
    # Access the model and tokenizer from app config
    model = current_app.config['MODEL']
    tokenizer = current_app.config['TOKENIZER']

    dataset = make_contrastive_dataset(synonyms, antonyms, suffix_list, template)
    model.reset()
    control_vector = ControlVector.train(model, tokenizer, dataset)
    return control_vector

################################################
# Main Endpoints 
################################################

# Endpoint to create a steerable model
@app.route('/steerable-model', methods=['POST'])
def create_steerable_model():
    try:
        data = request.get_json()
        model_label = data.get('model_label')
        control_dimensions = data.get('control_dimensions')
        suffix_list = data.get('suffix_list', DEFAULT_SUFFIX_LIST) 

        # Validate required fields
        if not model_label or not control_dimensions:
            app.logger.warning('Invalid request: missing model_label or control_dimensions')
            return jsonify({'error': 'model_label and control_dimensions are required'}), 400

        # Generate a 4-character unique identifier
        unique_id = uuid.uuid4().hex[:4]

        # Create the steering model name
        steering_model_full_id = f"{model_label}-{unique_id}"

        # Prepare to store control vectors for each dimension
        control_vectors = {}
        model_name = current_app.config['MODEL_NAME']

        # Create control vectors for each dimension
        for trait, (synonyms, antonyms) in control_dimensions.items():
            control_vectors[trait] = create_dataset_and_train_vector(synonyms, antonyms, suffix_list)

        # Store the steerable model with its control vectors
        steerable_model_with_vectors = {
            'id': steering_model_full_id,
            'object': 'steerable_model',
            'created_at': datetime.datetime.utcnow().isoformat(),
            'model': model_name,
            'control_vectors': control_vectors  # Stored internally, not returned to the user
        }

        steerable_models_vector_storage[steering_model_full_id] = steerable_model_with_vectors

        response = {
            'id': steering_model_full_id,
            'object': 'steerable_model',
            'created_at': steerable_model_with_vectors['created_at'],
            'model': model_name
        }

        app.logger.info('Steerable model created', extra={'model_id': steering_model_full_id})
        return jsonify(response), 201

    except Exception as e:
        app.logger.error('Error in create_steerable_model', extra={'error': str(e), 'traceback': traceback.format_exc()})
        return jsonify({'error': 'An internal error occurred', 'details': str(e)}), 500

# Endpoint to list steerable models
@app.route('/steerable-model', methods=['GET'])
def list_steerable_models():
    limit = request.args.get('limit', default=10, type=int)
    offset = request.args.get('offset', default=0, type=int)

    # Get the list of steerable models
    models_list = list(steerable_models_vector_storage.values())
    # Apply pagination
    models_list = models_list[offset:offset+limit]

    # Prepare the response data without control vectors
    data = [{
        'id': model['id'],
        'object': model['object'],
        'created_at': model['created_at'],
        'model': model['model'],
    } for model in models_list]

    app.logger.info('Steerable models listed', extra={'limit': limit, 'offset': offset})
    return jsonify({'data': data}), 200

# Endpoint to retrieve a specific steerable model
@app.route('/steerable-model/<model_id>', methods=['GET'])
def get_steerable_model(model_id):
    model = steerable_models_vector_storage.get(model_id)
    if not model:
        app.logger.warning('Steerable model not found', extra={'model_id': model_id})
        return jsonify({'error': 'Steerable model not found'}), 404

    # Return the model details without control vectors
    response = {
        'id': model['id'],
        'object': model['object'],
        'created_at': model['created_at'],
        'model': model['model'],
    }
    app.logger.info('Steerable model retrieved', extra={'model_id': model_id})
    return jsonify(response), 200

# Endpoint to delete a steerable model
@app.route('/steerable-model/<model_id>', methods=['DELETE'])
def delete_steerable_model(model_id):
    if model_id in steerable_models_vector_storage:
        del steerable_models_vector_storage[model_id]
        app.logger.info('Steerable model deleted', extra={'model_id': model_id})
        return jsonify({
            'id': model_id,
            'object': 'steerable_model',
            'deleted': True
        }), 200
    else:
        app.logger.warning('Steerable model not found for deletion', extra={'model_id': model_id})
        return jsonify({'error': 'Steerable model not found'}), 404

@app.route('/completions', methods=['POST'])
def generate_completion():
    try:
        data = request.get_json()
        model_name_request = data.get('model')
        prompt = data.get('prompt')
        control_settings = data.get('control_settings', {})
        generation_settings = data.get('settings', {})
    
        # Validate required fields
        if not model_name_request or not prompt:
            app.logger.warning('Invalid request: missing model or prompt')
            return jsonify({'error': 'Model and prompt are required'}), 400

        # Log input data
        app.logger.info('Completion requested', extra={'model': model_name_request, 'prompt': prompt})

        # Access the model and tokenizer from app config
        model = current_app.config['MODEL']
        tokenizer = current_app.config['TOKENIZER']
    
        # Prepare the input
        input_text = f"{user_tag}{prompt}{asst_tag}"
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
        # Move input_ids to the same device as the model
        input_ids = input_ids.to(next(model.parameters()).device)
    
        # Check if the model name corresponds to a model saved in local storage
        steerable_model = steerable_models_vector_storage.get(model_name_request)
        if steerable_model:
            control_vectors = steerable_model['control_vectors']

            # Start with a zero vector using the zero_like 
            matching_zero_vector = create_vector_with_zeros_like(next(iter(control_vectors.values())))

            vector_mix = sum(
                (control_vectors[trait] * control_settings.get(trait, 0.0) for trait in control_vectors),
                start=matching_zero_vector
            )
    
            # Apply the control vector to the model, if nonzero 
            if any(control_settings.get(trait, 0.0) != 0.0 for trait in control_vectors):
                model.set_control(vector_mix)  # This updates the model's control vectors
                app.logger.info('Control vector applied', extra={'control_settings': control_settings})
            else:
                app.logger.info(f"No control vectors found matching settings. Using base model for generation", extra={'model': model_name_request})
                model.reset()
        else:
            # Use the base model (no control vectors)
            app.logger.info(f"No prior model found for name '{model_name_request}'. Using base model for generation")
            model.reset()
    
        # Generation settings
        default_settings = {
            "do_sample": False,
            "max_new_tokens": 256,
            "repetition_penalty": 1.1,
        }
        generation_settings = {**default_settings, **generation_settings}
    
        # Log generation settings
        app.logger.debug('Generation settings', extra={'generation_settings': generation_settings})
    
        # Generate the output
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, **generation_settings)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Parse and format the generated text
        formatted_response = parse_assistant_response(generated_text)
    
        # Reset the model control after generation
        model.reset()
    
        # Prepare the response
        response = {
            'id': uuid.uuid4().hex,
            'object': 'text_completion',
            'created': datetime.datetime.utcnow().isoformat(),
            'model': model_name_request,
            'content': formatted_response,
        }
    
        app.logger.info('Completion generated', extra={'response_id': response['id']})
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