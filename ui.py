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

    for word, value in control_dimensions_dict.items():
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
    Displays the saved models with SELECT buttons for selection
    and properly formatted Control Dimensions.

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
            with st.expander(f"### Model:   **{model['model_name']}**", expanded=is_selected): # Model Name 
                control_dimensions = model.get('control_dimensions', {})

                if control_dimensions:
                    for word, examples in control_dimensions.items():
                        st.markdown(f"**{word}**")

                        positive_examples = examples.get('positive_examples', [])
                        negative_examples = examples.get('negative_examples', [])

                        if positive_examples:
                            st.markdown(f"**(+):** {', '.join(positive_examples)}")
                        else:
                            st.markdown("**(+):** _None provided_")

                        if negative_examples:
                            st.markdown(f"**(-):** {', '.join(negative_examples)}")
                        else:
                            st.markdown("**(-):** _None provided_")
                        st.markdown("---")  # Separator between control words
                else:
                    st.markdown("**No control dimensions provided for this model.**")

                # Highlight if this model is currently selected
                if is_selected:
                    st.success("**This model is currently selected.**")

def steer_model_page():
    st.title("Steer Model")

    st.markdown("---")

    st.header("Create Steerable Model")

    # Initialize session state for dynamic control dimensions
    if 'num_control_dimensions' not in st.session_state:
        st.session_state.num_control_dimensions = 3

    # Create column headers
    header_cols = st.columns([1, 1, 2])
    with header_cols[0]:
        st.write("**Control Word**")
        st.write("The type of behavior you want to steer.")
    with header_cols[2]:
        st.write("**Control Dimensions**")
        st.write('Click "Generate Examples" to generate a list of positive and negative examples for the control word, or enter your own list, formatted as JSON. ')

    placeholder_words = ["mercenary", "savvy", "sophisticated"]
    
    # Create dynamic rows for control dimensions
    for i in range(st.session_state.num_control_dimensions):
        if f'word_{i}' not in st.session_state:
            st.session_state[f'word_{i}'] = 'mercenary'
        if f'control_dimensions_{i}' not in st.session_state:
            default_json = json.dumps({
                "positive_examples": ["example1", "example2"],
                "negative_examples": ["example3", "example4"]
            })
            st.session_state[f'control_dimensions_{i}'] = default_json

        row_cols = st.columns([1, 1, 2])
        with row_cols[0]:
            if i < len(placeholder_words):
                placeholder_word = placeholder_words[i]
            else:
                placeholder_word = ""
            st.text_input(
                label=f"Control Word {i+1}",
                key=f"word_{i}",
                value=placeholder_word,
                label_visibility="collapsed"
            )
        with row_cols[1]:
            if st.button("Generate examples", key=f"generate_btn_{i}"):
                if st.session_state[f'word_{i}']:
                    result = generate_positive_negative_examples(st.session_state[f'word_{i}'])
                    if result:
                        try:
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
            st.text_area(
                label=f"Control Dimensions {i+1}",
                placeholder=placeholder_text,
                key=f"control_dimensions_text_{i}",
                height=200,
                label_visibility="collapsed",
                value=st.session_state[f'control_dimensions_{i}']
            )

    # Button to add new control dimension
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("➕ Add Word"):
            if 'num_control_dimensions' not in st.session_state:
                st.session_state.num_control_dimensions = 3  # Default value
            st.session_state.num_control_dimensions += 1
            st.rerun()
    with col2:
        if st.button("➖ Remove Word"):
            if 'num_control_dimensions' not in st.session_state:
                st.session_state.num_control_dimensions = 3  # Default value
            if st.session_state.num_control_dimensions > 1:  # Prevent removing all words
                st.session_state.num_control_dimensions -= 1
                st.rerun()

    # Text box to input the model name
    st.markdown("#### Name this model")
    model_name = st.text_input("", key="model_name", label_visibility='hidden' )

    # "Create Model" button
    if st.button("Create Model"):
        if not model_name:
            st.error("Please enter a model name.")
        else:
            control_dimensions = {}

            for i in range(st.session_state.num_control_dimensions):
                control_word = st.session_state.get(f'word_{i}', '').strip()
                control_input = st.session_state.get(f'control_dimensions_{i}', '').strip()

                if control_word and control_input:
                    try:
                        parsed_json = json.loads(control_input)
                        if not isinstance(parsed_json, dict):
                            raise ValueError(f"Control dimension '{control_word}' is not a JSON object.")
                        if "positive_examples" not in parsed_json or "negative_examples" not in parsed_json:
                            raise ValueError(f"Control dimension '{control_word}' must contain 'positive_examples' and 'negative_examples' keys.")
                        control_dimensions[control_word] = parsed_json
                    except json.JSONDecodeError as e:
                        st.warning(f"Invalid JSON format in control dimension '{control_word}': {str(e)}")
                    except ValueError as ve:
                        st.warning(str(ve))

            if control_dimensions:
                # Get current timestamp
                timestamp = int(datetime.now().timestamp())
                # Generate a unique ID combining timestamp and model name
                steering_model_full_id = f"{timestamp}_{model_name}"

                # Get current timestamp in ISO format
                created_at = datetime.now().isoformat()

                control_dimension_words = list(control_dimensions.keys())

                # Create the model dictionary
                api_response = {
                    'id': steering_model_full_id,
                    'model_name': model_name,
                    'created_at': created_at,
                    'control_dimensions': control_dimensions,
                    'control_dimension_words': control_dimension_words
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

    # Third section: Generate
    st.markdown("---")
    st.markdown("### Generate")

    if st.session_state.get('current_model') is None:
        st.warning("Please select a model from the 'Saved Models' section above.")
    else:
        # Display the selected model ID
        selected_model_id = st.session_state['current_model']
        st.write(f"**Generating with Model ID:** `{selected_model_id}`")

        # Load the model details from models.json
        try:
            with open("models.json", "r") as f:
                models_data = json.load(f)
                # Find the selected model
                selected_model = next((model for model in models_data if model['id'] == selected_model_id), None)
                if selected_model is None:
                    st.error("Selected model not found in saved models.")
                    return
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return

        # Sliders for control dimensions
        control_settings = {}
        control_dimensions = selected_model.get('control_dimensions', {})

        if control_dimensions:
            st.markdown("#### Adjust Control Dimensions")
            for word in control_dimensions.keys():
                col1, col2 = st.columns([1, 5])
                
                # Number input
                with col1:
                    number_value = st.number_input(
                        label=f"{word}",
                        min_value=-10,
                        max_value=10,
                        value=0,
                        step=1,
                        key=f"number_{word}"
                    )
                
                # Slider
                with col2:
                    slider_value = st.slider(
                        label=f"",
                        min_value=-10,
                        max_value=10,
                        value=int(number_value),
                        key=f"slider_{word}"
                    )
                
                # Synchronize number input and slider
                if slider_value != number_value:
                    st.session_state[f"number_{word}"] = slider_value
                
                control_settings[word] = slider_value
        else:
            st.info("This model has no control dimensions.")

        # Text box for user_prompt
        user_prompt = st.text_input("Enter your prompt:", value="How do you think about the universe?", key="user_prompt")

        # Button to generate
        if st.button("Generate Response"):
            if not user_prompt:
                st.error("Please enter a prompt.")
            else:
                # Prepare the API request
                MODEL_SERVER_URL = "https://example.com/api"  # Replace with actual URL

                payload = {
                    "model": selected_model_id,
                    "prompt": user_prompt,
                    "control_settings": control_settings,
                    # Include any other necessary fields
                }

                # Display the payload being sent
                st.markdown("**API Request Payload:**")
                st.json(payload)

                # Make the API request - for now, we can simulate this
                try:
                    # Simulated response
                    result = {"content": "This is a sample response based on your input and control settings. It would be much longer in a real scenario, potentially including multiple paragraphs or even pages of generated text, depending on the complexity of the prompt and the capabilities of the model."}

                    # Save to session state
                    st.session_state['model_response'] = result.get("content", "No content received from the model.")
                except Exception as e:
                    st.error(f"Error making API request: {str(e)}")
                    st.session_state['model_response'] = f"Error: {str(e)}"

        # Large text box for response
        st.markdown("#### **Response:**")
        st.text_area(
            label="Model Response",
            value=st.session_state.get('model_response', "No response generated yet. Click 'Generate Response' to get a response."),
            height=300,
            key="model_response_display"
        )

    # New section: Testing Arena
    st.markdown("---")
    st.markdown("### Testing Arena")


    # List of prompts
    default_prompts = """Write a story about a brave knight.

Describe a futuristic city.

Explain the process of photosynthesis.

Write a recipe for chocolate chip cookies.

Discuss the impact of social media on society."""

    test_prompts = st.text_area("List of Test Prompts (separated by double-lines)", value=default_prompts, height=200, key="test_prompts")

    # Testing prompt
    if 'control_dimensions' in selected_model:
        default_testing_prompt = f"Please act like this: {json.dumps(selected_model['control_dimensions'])}"
    else:
        default_testing_prompt = "Please act according to the following instructions:"

    testing_prompt = st.text_area("Control Prompt", value=default_testing_prompt, height=100, key="testing_prompt")

    # Process and display results for each test prompt
    prompts = [p.strip() for p in test_prompts.split('\n\n') if p.strip()]

    # Add a button to generate the test suite
    if st.button("Generate Test Suite"):
        st.info("Test suite generation initiated. This may take a moment...")
        # Here you would typically call a function to generate the test suite
        # For now, we'll just display a placeholder message
        st.success("Test suite generated successfully!")

    st.markdown("---")
    st.markdown(f"\n#### Responses:")
    for prompt in prompts:
        st.markdown(f"**{prompt}**")

        col1, col2, col3, col4= st.columns(4)

        with col1:
            st.markdown("Standard")
            response = "Sample response for prompt only."  # Replace with actual API call
            st.text_area("Response", value=response, height=150, key=f"standard_response_{prompt[:20]}")

        with col2:
            st.markdown("With Prompt")
            response = "Sample response with positive control vectors."  # Replace with actual API call
            st.text_area("Response", value=response, height=150, key=f"prompt_response_{prompt[:20]}")

        with col3:
            st.markdown("(+) Control Vectors")
            response = "Sample response with negative control vectors."  # Replace with actual API call
            st.text_area("Response", value=response, height=150, key=f"response_positive_{prompt[:20]}")

        with col4:
            st.markdown("(-) Vectors (Inverted)")
            response = "Sample response with negative control vectors."  # Replace with actual API call
            st.text_area("Response", value=response, height=150, key=f"response_negative_{prompt[:20]}")

        st.markdown("---")

def main():
    st.set_page_config(page_title="Steerable Models App", layout="wide")
    steer_model_page()

if __name__ == "__main__":
    main()

    # TODO: add something to check the list from the API to ensure the model is valid before calling it. update the list with 'model is valid' beforehand 
