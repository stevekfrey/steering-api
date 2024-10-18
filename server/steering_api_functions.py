import os
import uuid
import datetime
import warnings
import torch
import numpy as np
import logging
import json
import threading
from enum import Enum
from dotenv import load_dotenv
from huggingface_hub import login
from typing import Callable, Any, Dict
from tqdm import tqdm
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel, DatasetEntry

from steer_templates import DEFAULT_TEMPLATE, DEFAULT_PROMPT_LIST, user_tag, asst_tag, BASE_MODEL_NAME

# Initialize a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

################################################
# Model Loading and Initialization
################################################

def load_model():
    """
    Loads and initializes the base model and tokenizer.
    Returns:
        model (ControlModel): The initialized model.
        tokenizer (AutoTokenizer): The tokenizer.
    """
    warnings.filterwarnings('ignore')
    torch.cuda.empty_cache()

    # Load environment variables
    load_dotenv()

    # Login to Hugging Face
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if hf_token:
        login(token=hf_token)
    else:
        logger.warning("No Hugging Face token found in .env file. You may encounter issues accessing models.")

    model_name = BASE_MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        token=hf_token  # Use the token here
    )

    logger.info(f"Loaded base model {model_name}")

    # Wrap the model with ControlModel
    control_layers = list(range(-5, -18, -1))
    model = ControlModel(base_model, control_layers)

    return model, tokenizer

################################################
# In-Memory Storage for Steerable Models
################################################

steerable_models_vector_storage = {}

def get_steerable_model(model_id):
    # First, check in-memory storage
    if model_id in steerable_models_vector_storage:
        print(f"Found {model_id} in memory")
        return steerable_models_vector_storage[model_id]

    # If not found in memory, check the JSON file
    filename = "model_steering_data.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            models_list = json.load(f)
        for model_data in models_list:
            if model_data['id'] == model_id:
                # Convert the control vectors to the correct format
                model_data['control_vectors'] = {
                    trait: ControlVector(
                        model_type=cv['model_type'],
                        directions={int(k): torch.tensor(v) for k, v in cv['directions'].items()}
                    )
                    for trait, cv in model_data['control_vectors'].items()
                }
                # Store it in memory for faster future access
                steerable_models_vector_storage[model_id] = model_data
                print(f"Found {model_id} in JSON file and loaded into memory")
                return model_data
    
    # Model not found
    return None

def list_steerable_models(limit=10, offset=0):
    models_list = list(steerable_models_vector_storage.values())
    return models_list[offset:offset+limit]

def save_steerable_model(model_data):
    model_id = model_data['id']
    steerable_models_vector_storage[model_id] = model_data

def delete_steerable_model(model_id):
    if model_id in steerable_models_vector_storage:
        del steerable_models_vector_storage[model_id]
        return True
    return False

################################################
# Data Helpers and Processing
################################################


def load_prompt_list(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def make_contrastive_dataset(
    positive_personas: list,
    negative_personas: list,
    prompt_list: list,
    template: str = DEFAULT_TEMPLATE,
    tokenizer = None
) -> list:
    """
    Creates a contrastive dataset from positive and negative personas.

    Args:
        positive_personas (list): List of positive persona strings.
        negative_personas (list): List of negative persona strings.
        prompt_list (list): List of prompt strings.
        template (str): Template string for dataset entries.
        tokenizer: Tokenizer for processing text.

    Returns:
        dataset (list): List of DatasetEntry objects.
    """
    logger.info(f"Making contrastive dataset with {len(positive_personas)} positive personas, {len(negative_personas)} negative personas, and {len(prompt_list)} test prompts to generate data ")

    dataset = []
    for prompt in prompt_list:
        tokens = tokenizer.tokenize(prompt)
        for i in range(1, len(tokens)):
            truncated_prompt = tokenizer.convert_tokens_to_string(tokens[:i])
            for positive_persona, negative_persona in zip(positive_personas, negative_personas):
                positive_template = template.format(user_tag=user_tag, asst_tag=asst_tag, persona=positive_persona, prompt=truncated_prompt)
                negative_template = template.format(user_tag=user_tag, asst_tag=asst_tag, persona=negative_persona, prompt=truncated_prompt)
                dataset.append(
                    DatasetEntry(
                        positive=f"{user_tag} {positive_template} {asst_tag} {truncated_prompt}",
                        negative=f"{user_tag} {negative_template} {asst_tag} {truncated_prompt}",
                    )
                )
    
    logger.info(f"Created contrastive dataset with {len(dataset)} entries")
    return dataset

def parse_assistant_response(full_string):
    """
    Parses the assistant's response from the model's output.

    Args:
        full_string (str): The raw output string from the model.

    Returns:
        response (str): The parsed assistant response.
    """
    parts = full_string.split("Assistant:")
    if len(parts) > 1:
        response = parts[-1].strip()
        response = response.rstrip("").strip()
        return response
    return ""

def create_vector_with_zeros_like(control_vector):
    """
    Creates a zero vector matching the shape of the given control vector.

    Args:
        control_vector (ControlVector): The control vector to match.

    Returns:
        ControlVector: A new ControlVector with zero directions.
    """
    zero_directions = {k: torch.zeros_like(torch.tensor(v)) if isinstance(v, np.ndarray) else torch.zeros_like(v)
                       for k, v in control_vector.directions.items()}
    return ControlVector(model_type=control_vector.model_type, directions=zero_directions)

def create_dataset_and_train_vector(positive_examples, negative_examples, prompt_list, template, model, tokenizer):
    """
    Creates a dataset and trains a control vector.

    Args:
        positive_examples (list): List of positive persona strings.
        negative_examples (list): List of negative persona strings.
        prompt_list (list): List of prompt strings.
        template (str): Template string for dataset entries.
        model (ControlModel): The model to train.
        tokenizer: Tokenizer for processing text.

    Returns:
        control_vector (ControlVector): The trained control vector.
    """
    logger.info(f"Creating dataset with {len(positive_examples)} positive examples and {len(negative_examples)} negative examples")
    dataset = make_contrastive_dataset(positive_examples, negative_examples, prompt_list, template, tokenizer)
    logger.info(f"Dataset created with {len(dataset)} entries")

    if len(dataset) == 0:
        logger.error("Dataset is empty. Cannot train control vector.")
        raise ValueError("Empty dataset")

    model.reset()
    control_vector = ControlVector.train(model, tokenizer, dataset)
    return control_vector

################################################
# Completion Generation
################################################

def generate_completion_response(
    model_name_request,
    prompt,
    control_settings,
    generation_settings,
    model,
    tokenizer,
    control_vectors=None
):
    """
    Generates a completion response based on the input prompt and control settings.

    Args:
        model_name_request (str): The name of the model to use.
        prompt (str): The input prompt.
        control_settings (dict): Settings for control dimensions.
        generation_settings (dict): Settings for text generation.
        model (ControlModel): The model to use.
        tokenizer: Tokenizer for processing text.
        control_vectors (dict): Preloaded control vectors.

    Returns:
        response (dict): The generated response.
    """
    print(f"Received control_settings in generate_completion_response: {control_settings}")

    input_text = f"{user_tag}{prompt}{asst_tag}"
    encoding = tokenizer.encode_plus(
        input_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    # Move inputs to the model's device
    device = next(model.parameters()).device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Load control vectors if not provided
    if control_vectors is None:
        steerable_model = get_steerable_model(model_name_request)
        if steerable_model:
            control_vectors = steerable_model['control_vectors']
        else:
            control_vectors = {}

    if control_vectors:
        # Ensure control vectors are correctly deserialized and tensors are properly formatted
        for trait, cv in control_vectors.items():
            for layer_key in cv.directions:
                # If the direction is not a tensor, convert it
                if not isinstance(cv.directions[layer_key], torch.Tensor):
                    cv.directions[layer_key] = torch.tensor(cv.directions[layer_key])

        # Update model.control_layers to match control vector layers
        control_layers = list(next(iter(control_vectors.values())).directions.keys())
        model.control_layers = control_layers

        # Create a zero vector matching the control vector dimensions
        matching_zero_vector = create_vector_with_zeros_like(next(iter(control_vectors.values())))

        # Compute the weighted sum of control vectors based on control settings
        vector_mix = sum(
            (
                control_vectors[trait] * control_settings.get(trait, 0.0)
                for trait in control_vectors
            ),
            start=matching_zero_vector
        )

        # print('vector_mix:', vector_mix)
        print('control_settings:', control_settings)
        # print('control_vectors:', control_vectors)

        # Apply the control vector if any trait has a non-zero setting
        if any(control_settings.get(trait, 0.0) != 0.0 for trait in control_vectors):
            model.set_control(vector_mix)
            logger.info('Control vector applied', extra={'control_settings': control_settings})
        else:
            logger.info("No control settings applied. Using base model for generation.")
            model.reset()
    else:
        # Use the base model if no control vectors are found
        logger.info(f"No control vectors found for model '{model_name_request}'. Using base model for generation.")
        model.reset()

    # Merge default generation settings with provided settings
    default_settings = {
        "do_sample": False,
        "max_new_tokens": 256,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.pad_token_id,
    }
    gen_settings = {**default_settings, **generation_settings}

    # Generate the output
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_settings
        )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Parse and format the generated text
    formatted_response = parse_assistant_response(generated_text)

    # Reset the model control after generation
    model.reset()

    # Prepare the response
    response = {
        'model_id': model_name_request,
        'object': 'text_completion',
        'created': datetime.datetime.utcnow().isoformat(),
        'model': model_name_request,
        'content': formatted_response,
    }

    logger.info('Completion generated', extra={'response_model_id': response['model_id']})
    return response

################################################
# Steerable Model Creation Function
################################################

class ModelStatus(Enum):
    CREATING = "creating"
    READY = "ready"
    FAILED = "failed"

model_status = {}

def create_steerable_model_function(
    model_id: str, 
    model_label: str,
    control_dimensions: dict,
    prompt_list: list,
    model,
    tokenizer,
    template: str = DEFAULT_TEMPLATE
):
    """
    Creates a steerable model with control vectors based on the provided control dimensions.
    
    Args:
        model_label (str): A label for the model.
        control_dimensions (dict): Dictionary with traits and their positive and negative examples.
        prompt_list (list): List of prompt strings to use in dataset creation.
        model (ControlModel): The model to train.
        tokenizer: Tokenizer for processing text.
        template (str): Template string for dataset entries.
    
    Returns:
        dict: A dictionary containing the steerable model information.
    """
    try:
        if not model_label or not control_dimensions:
            raise ValueError('model_label and control_dimensions are required')

        steering_model_full_id = model_id

        # Prepare to store control vectors for each dimension
        control_vectors = {}
        model_name = BASE_MODEL_NAME

        # Create control vectors for each dimension
        for trait, examples in control_dimensions.items():
            positive_examples = examples.get('positive_examples', [])
            negative_examples = examples.get('negative_examples', [])
            logger.info(f"Processing trait: {trait}")
            logger.info(f"Positive examples: {positive_examples}")
            logger.info(f"Negative examples: {negative_examples}")
            logger.info(f"Prompt list: {prompt_list}")
            control_vectors[trait] = create_dataset_and_train_vector(
                positive_examples, negative_examples, prompt_list, template, model, tokenizer
            )

        # Store the steerable model with its control vectors and control dimensions
        created_at = datetime.datetime.utcnow().isoformat()

        # --- Modify code to save control vectors to a single JSON file ---
        # Function to serialize ControlVector objects
        def serialize_control_vector(control_vector):
            directions_serializable = {
                k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in control_vector.directions.items()
            }
            return {
                'model_type': control_vector.model_type,
                'directions': directions_serializable
            }

        # Serialize control vectors
        serialized_control_vectors = {
            trait: serialize_control_vector(cv) for trait, cv in control_vectors.items()
        }

        # Prepare data to save
        steerable_model_data_to_save = {
            'id': steering_model_full_id,
            'object': 'steerable_model',
            'created_at': created_at,
            'model': model_name,
            'control_vectors': serialized_control_vectors,
            'control_dimensions': control_dimensions
        }

        # Save to a single JSON file
        filename = "model_steering_data.json"
        try:
            # Check if the file exists
            if os.path.exists(filename):
                # Load existing data
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            # Append the new model data
            existing_data.append(steerable_model_data_to_save)

            # Save back to the JSON file
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=4, default=str)
            logger.info(f"Steerable model data saved to {filename}")
        except Exception as json_error:
            logger.error(f"Error saving steerable model data to JSON: {str(json_error)}")
            raise

        # Prepare the response
        response = {
            'id': steering_model_full_id,
            'object': 'steerable_model',
            'created_at': created_at,  # Use the already formatted string
            'model': model_name,
            'control_dimensions': control_dimensions
        }

        logger.info('Steerable model created', extra={'model_id': steering_model_full_id, 'response': response})

        return response

    except Exception as e:
        logger.error('Error in create_steerable_model_function', extra={'error': str(e)})
        raise

def create_steerable_model_async(model_label, control_dimensions, prompt_list, model, tokenizer, template=DEFAULT_TEMPLATE):
    unique_id = f"{random.randint(10, 99)}"  
    steering_model_full_id = f"{model_label}-{unique_id}"
    
    model_status[steering_model_full_id] = ModelStatus.CREATING
    
    thread = threading.Thread(target=create_steerable_model_background, 
                              args=(steering_model_full_id, model_label, control_dimensions, prompt_list, model, tokenizer, template))
    thread.start()
    
    return {"id": steering_model_full_id, "status": ModelStatus.CREATING.value}

# def create_steerable_model_background(steering_model_full_id, model_label, control_dimensions, prompt_list, model, tokenizer, template):
#     try:
#         # Existing model creation logic here
#         # ...

#         model_status[steering_model_full_id] = ModelStatus.READY
#     except Exception as e:
#         logger.error(f"Error creating model {steering_model_full_id}: {str(e)}")
#         model_status[steering_model_full_id] = ModelStatus.FAILED

def get_model_status(model_id):
    return {"id": model_id, "status": model_status.get(model_id, ModelStatus.FAILED).value}
