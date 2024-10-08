from flask import Flask, request, jsonify
import uuid
import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel, DatasetEntry
import pickle
import warnings
import numpy as np

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()

torch.mps.empty_cache()

app = Flask(__name__)

# Load the model and tokenizer at startup
# model_name = "TinyLlama/TinyLlama-1.1B-step-50K-105b"
model_name = "aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored"

tokenizer = AutoTokenizer.from_pretrained(model_name)


# Add this line to set the pad token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

########################################
#### load from remote ####
# base_model = AutoModelForCausalLM.from_pretrained(model_name)
# base_model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
# model = AutoModelForCausalLM.from_pretrained("local_models", device_map="auto", local_files_only=True)
# base_model.save_pretrained("local_models", safe_serialization=True)

#### load from local ####
base_model = AutoModelForCausalLM.from_pretrained("local_models", device_map="auto", local_files_only=True)

print ("saved base model to local_models")
########################################


if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")

base_model = base_model.to("cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")
base_model.to(device)

# Wrap the model with ControlModel
control_layers = list(range(-5, -18, -1))
model = ControlModel(base_model, control_layers)
model.to(device)


user_tag = "<|start_header_id|>user<|end_header_id|>You: "
asst_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>Assistant:"
DEFAULT_TEMPLATE = "I am a {persona} person."

# In-memory storage for steerable models and their control vectors
steerable_models = {}



# Function to create a contrastive dataset
def make_contrastive_dataset(
    positive_personas: list,
    negative_personas: list,
    suffix_list: list,
    template: str = DEFAULT_TEMPLATE
) -> list:
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

# If ControlVector.zero_like() is not available, you can implement it or use an alternative method to create a zero control vector. Here's a possible implementation:
@staticmethod
def zeros_like(control_vector):
    zero_directions = {k: torch.zeros_like(torch.tensor(v)) if isinstance(v, np.ndarray) else torch.zeros_like(v) 
                       for k, v in control_vector.directions.items()}
    return ControlVector(model_type=control_vector.model_type, directions=zero_directions)



# Function to create dataset and train control vector
def create_dataset_and_train_vector(synonyms, antonyms, suffix_list, template=DEFAULT_TEMPLATE):
    dataset = make_contrastive_dataset(synonyms, antonyms, suffix_list, template)
    model.reset()
    control_vector = ControlVector.train(model, tokenizer, dataset)
    return control_vector

# Endpoint to create a steerable model
@app.route('/steerable-models', methods=['POST'])
def create_steerable_model():
    data = request.get_json()
    model_label = data.get('model_label')
    control_dimensions = data.get('control_dimensions')
    suffix_list = data.get('suffix_list', [''])  # Provide a default suffix list

    # Validate required fields
    if not model_label or not control_dimensions:
        return jsonify({'error': 'model_label and control_dimensions are required'}), 400

    # Generate a 4-character unique identifier
    unique_id = uuid.uuid4().hex[:4]

    # Create the steering model name
    steering_model = f"{model_label}-{unique_id}"

    # Prepare to store control vectors for each dimension
    control_vectors = {}

    # Create control vectors for each dimension
    for trait, (synonyms, antonyms) in control_dimensions.items():
        control_vector = create_dataset_and_train_vector(synonyms, antonyms, suffix_list)
        control_vectors[trait] = control_vector

    # Store the steerable model with its control vectors
    steerable_model = {
        'id': steering_model,
        'object': 'steerable_model',
        'created_at': datetime.datetime.utcnow().isoformat(),
        'model': model_name,
        'steering_model': steering_model,
        'control_vectors': control_vectors  # Stored internally, not returned to the user
    }

    # Save the steerable model using the full model ID as the key
    steerable_models[steering_model] = steerable_model

    # Return the response without control vectors
    response = {
        'id': steering_model,
        'object': 'steerable_model',
        'created_at': steerable_model['created_at'],
        'model': model_name,
        'steering_model': steering_model
    }

    return jsonify(response), 201

# Endpoint to list steerable models
@app.route('/steerable-models', methods=['GET'])
def list_steerable_models():
    limit = request.args.get('limit', default=10, type=int)
    offset = request.args.get('offset', default=0, type=int)

    # Get the list of steerable models
    models_list = list(steerable_models.values())

    # Apply pagination
    models_list = models_list[offset:offset+limit]

    # Prepare the response data without control vectors
    data = [{
        'id': model['id'],
        'object': model['object'],
        'created_at': model['created_at'],
        'model': model['model'],
        'steering_model': model['steering_model']
    } for model in models_list]

    return jsonify({'data': data}), 200

# Endpoint to retrieve a specific steerable model
@app.route('/steerable-models/<model_id>', methods=['GET'])
def get_steerable_model(model_id):
    model = steerable_models.get(model_id)
    if not model:
        return jsonify({'error': 'Steerable model not found'}), 404

    # Return the model details without control vectors
    response = {
        'id': model['id'],
        'object': model['object'],
        'created_at': model['created_at'],
        'model': model['model'],
        'steering_model': model['steering_model']
    }
    return jsonify(response), 200

# Endpoint to delete a steerable model
@app.route('/steerable-models/<model_id>', methods=['DELETE'])
def delete_steerable_model(model_id):
    if model_id in steerable_models:
        del steerable_models[model_id]
        return jsonify({
            'id': model_id,
            'object': 'steerable_model',
            'deleted': True
        }), 200
    else:
        return jsonify({'error': 'Steerable model not found'}), 404

# Endpoint to generate a text completion
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

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
            return jsonify({'error': 'Model and prompt are required'}), 400
    
        # Log input data
        logging.debug(f"Input data: {data}")
    
        # Prepare the input
        input_text = f"{user_tag}{prompt}{asst_tag}"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
        # Check if the model name corresponds to a steerable model
        steerable_model = steerable_models.get(model_name_request)
        if steerable_model:
            # Retrieve the control vectors
            control_vectors = steerable_model['control_vectors']
    
            # Build the combined control vector
            # Start with a zero vector using the zero_like function you defined
            zero_vector = zeros_like(next(iter(control_vectors.values())))
            vector_mix = sum(
                (control_vectors[trait] * control_settings.get(trait, 0.0) for trait in control_vectors),
                start=zero_vector
            )
    
            # Apply the control vector to the model if any control is applied
            if any(control_settings.get(trait, 0.0) != 0.0 for trait in control_vectors):
                model.set_control(vector_mix)
            else:
                model.reset()
        else:
            # Use the base model (no control vectors)
            model.reset()
    
        # Generation settings
        default_settings = {
            # "pad_token_id": tokenizer.pad_token_id,
            # "eos_token_id": tokenizer.eos_token_id,
            "do_sample": False,
            "max_new_tokens": 256,
            "repetition_penalty": 1.1,
        }
        generation_settings = {**default_settings, **generation_settings}
    
        # Log generation settings
        logging.debug(f"Generation settings: {generation_settings}")
    
        # Generate the output
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, **generation_settings)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
        # Reset the model control after generation
        model.reset()
    
        # Prepare the response
        response = {
            'id': uuid.uuid4().hex,
            'object': 'text_completion',
            'created': datetime.datetime.utcnow().isoformat(),
            'model': model_name_request,
            'choices': [
                {
                    'text': generated_text,
                    'index': 0,
                    'logprobs': None,
                    'finish_reason': 'stop'
                }
            ]
        }
    
        return jsonify(response), 200
    
    except Exception as e:
        logging.error(f"Error in generate_completion: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': 'An internal error occurred', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
