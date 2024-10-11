import streamlit as st
import json
import os
from dotenv import load_dotenv
from litellm import completion
from datetime import datetime  # Added import for timestamp

# Load environment variables
load_dotenv()

# Initialize session state for current_model if not present
if 'current_model' not in st.session_state:
    st.session_state['current_model'] = None

def generate_positive_negative_examples(word):
    prompt = f"""Here is a word, please create an ordered list of 5 SYNONYMS (similar to the word) and then 5 ANTONYMS (the opposite of the word). Respond ONLY with a JSON object containing two keys: "positive_examples" and "negative_examples". Each key should map to a list of 5 examples. Respond with the following format:

    =====
    EXAMPLE INPUT:
    materialistic

    EXAMPLE RESPONSE:
    {{
        "positive_examples": ["materialistic", "consumerist", "acquisitive", "wealthy", "greedy"],
        "negative_examples": ["minimalist", "austere", "spiritual", "altruistic", "ascetic"]
    }}
    =====
    INPUT:
    {word}

    RESPONSE:
"""

    try:
        response = completion(
            model="anthropic/claude-3-haiku-20240307",
            messages=[{"role": "user", "content": prompt}],
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generating positive and negative examples: {str(e)}")
        return None

def parse_control_dimensions(control_dimensions_dict):
    """
    Parses a dictionary containing control dimensions.

    Args:
        control_dimensions_dict (dict): The dictionary containing control dimensions.

    Returns:
        tuple: A tuple containing two lists: positive_examples and negative_examples.
    """
    positive_examples = []
    negative_examples = []

    for key, value in control_dimensions_dict.items():
        if isinstance(value, dict):
            pos = value.get('positive_examples', [])
            neg = value.get('negative_examples', [])
            positive_examples.extend(pos)
            negative_examples.extend(neg)

    return positive_examples, negative_examples

def select_model(model_id):
    """
    Callback function to set the selected model in session_state.

    Args:
        model_id (str): The ID of the model to select.
    """
    st.session_state['current_model'] = model_id

def display_saved_models(saved_models):
    """
    Displays the saved models with SELECT buttons for selection and properly formatted Control Dimensions.

    Args:
        saved_models (list of dict): A list where each dict contains model details,
                                     including 'control_dimensions' dictionary.
    """

    if not saved_models:
        st.write("No models saved yet.")
        return

    for model in saved_models:
        cols = st.columns([0.15, 1])  # First column ~15% width for the SELECT button

        with cols[0]:
            # Determine if this model is currently selected
            is_selected = st.session_state.get('current_model') == model['id']

            # Define button label based on selection
            button_label = "✅ Current Model" if is_selected else "🔲 Select model"

            # The 'disabled' parameter visually indicates selection by disabling the button
            st.button(
                button_label,
                key=f"select_{model['id']}",
                on_click=select_model,
                args=(model['id'],),
                disabled=is_selected,  # Disable if selected
            )

        with cols[1]:
            # Expand the expander if the model is selected
            with st.expander(f"### **Model Name: {model['model_name']}**", expanded=is_selected):
                control_dimensions_dict = model.get('control_dimensions', {})
                positive_examples, negative_examples = parse_control_dimensions(control_dimensions_dict)
                
                if positive_examples:
                    st.markdown(f"**Positive Examples:** {', '.join(positive_examples)}")
                else:
                    st.markdown("**Positive Examples:** _None provided_")
                
                if negative_examples:
                    st.markdown(f"**Negative Examples:** {', '.join(negative_examples)}")
                else:
                    st.markdown("**Negative Examples:** _None provided_")
                
                # Highlight if this model is currently selected
                if is_selected:
                    st.success("**This model is currently selected.**")

def steer_model_page():
    st.title("Steer Model")

    st.header("Create Steerable Model")

    # Initialize session state for text inputs and text areas
    for i in range(3):
        if f'word_{i}' not in st.session_state:
            st.session_state[f'word_{i}'] = 'mercenary'
        if f'control_dimensions_{i}' not in st.session_state:
            # Initialize with a compact default JSON string
            default_json = json.dumps({
                "positive_examples": ["example1", "example2"],
                "negative_examples": ["example3", "example4"]
            })
            st.session_state[f'control_dimensions_{i}'] = default_json

    # Create column headers
    header_cols = st.columns([1, 1, 2])
    with header_cols[0]:
        st.write("**Control Word**")
        st.write("The type of behavior you want to steer.")
    with header_cols[2]:
        st.write("**Control Dimensions**")
        st.write('Click "Generate Examples" to generate a list of positive and negative examples for the control word, or enter your own list.')

    # Create 3 rows with text input, generate button, and control dimensions text area
    for i in range(3):
        row_cols = st.columns([1, 1, 2])
        with row_cols[0]:
            st.text_input(
                label=f"control_word_{i}",
                key=f"word_{i}",
                placeholder="",
                label_visibility='collapsed'
            )
        with row_cols[1]:
            # Use a unique button key for each button
            if st.button("Generate examples", key=f"generate_btn_{i}"):
                if st.session_state[f'word_{i}']:
                    result = generate_positive_negative_examples(st.session_state[f'word_{i}'])
                    if result:
                        try:
                            # Ensure the result is a compact JSON string
                            if isinstance(result, dict):
                                result = json.dumps(result)
                            st.session_state[f'control_dimensions_{i}'] = result
                            st.success(f"Examples generated for control word {i+1}.")
                        except (TypeError, json.JSONDecodeError) as e:
                            st.error(f"Failed to parse generated examples: {str(e)}")
        with row_cols[2]:
            placeholder_text = (
                '{'
                '"positive_examples": ["example1", "example2"], '
                '"negative_examples": ["example3", "example4"]'
                '}'
            )
            user_input = st.text_area(
                label=f"word_examples_{i}",
                placeholder=placeholder_text,
                key=f"control_dimensions_text_{i}",
                height=200,
                label_visibility='collapsed',
                value=st.session_state[f'control_dimensions_{i}']
            )
            # Update the session state with the user's input (as a string)
            st.session_state[f'control_dimensions_{i}'] = user_input

    # Text box to input the model name
    model_name = st.text_input("Name this model", key="model_name")

    # "Create Model" button
    if st.button("Create Model"):
        if not model_name:
            st.error("Please enter a model name.")
        else:
            control_dimensions = {}
            # Removed parsing_errors flag

            for i in range(3):
                control_input = st.session_state.get(f'control_dimensions_{i}', '').strip()
                if not control_input:
                    continue  # Skip if empty
                try:
                    parsed_json = json.loads(control_input)
                    # Validate the JSON structure
                    if not isinstance(parsed_json, dict):
                        raise ValueError(f"Control dimension {i+1} is not a JSON object.")
                    if "positive_examples" not in parsed_json or "negative_examples" not in parsed_json:
                        raise ValueError(f"Control dimension {i+1} must contain 'positive_examples' and 'negative_examples' keys.")
                    control_dimensions[f'control_dimension_{i}'] = parsed_json
                except json.JSONDecodeError as e:
                    st.warning(f"Invalid JSON format in control dimension {i+1}: {str(e)}")
                    # Skip invalid entries
                except ValueError as ve:
                    st.warning(str(ve))
                    # Skip invalid entries

            if control_dimensions:
                # Get current timestamp
                timestamp = int(datetime.now().timestamp())
                # Generate a unique ID combining timestamp and model name
                steering_model_full_id = f"{timestamp}_{model_name}"

                # Get current timestamp in ISO format
                created_at = datetime.now().isoformat()

                # Create the model dictionary
                api_response = {
                    'id': steering_model_full_id,
                    'model_name': model_name,
                    'created_at': created_at,
                    'control_dimensions': control_dimensions 
                }

                # Save the response locally in a JSON file
                try:
                    models_data = []
                    if os.path.exists("models.json"):
                        with open("models.json", "r") as f:
                            try:
                                models_data = json.load(f)
                            except json.JSONDecodeError:
                                st.error("Error reading the models.json file. Please ensure it's a valid JSON.")
                                models_data = []

                    # Append new model
                    models_data.append(api_response)

                    # Save back to the file
                    with open("models.json", "w") as f:
                        json.dump(models_data, f, indent=4)

                    st.success("Model created and saved locally.")
                except Exception as e:
                    st.error(f"Failed to save the model: {str(e)}")
            else:
                st.warning("No valid control dimensions provided. Model not saved.")

    # "Reset Models" Button
    st.markdown("---")  # Separator

    # Display saved models
    st.markdown("### Saved Models")

    if st.button("Reset Models"):
        try:
            with open("models.json", "w") as f:
                json.dump([], f, indent=4)
            st.success("All models have been reset. 'models.json' is now empty.")
            st.session_state['current_model'] = None  # Clear the current_model selection
        except Exception as e:
            st.error(f"Failed to reset models: {str(e)}")
    if os.path.exists("models.json"):
        with open("models.json", "r") as f:
            try:
                models_data = json.load(f)
                if models_data:
                    display_saved_models(models_data)
                else:
                    st.write("No models saved yet.")
            except json.JSONDecodeError:
                st.error("Error reading the models.json file. Please ensure it's a valid JSON.")
    else:
        st.write("No models saved yet.")

def main():
    st.set_page_config(page_title="Steerable Models App")
    steer_model_page()

if __name__ == "__main__":
    main()