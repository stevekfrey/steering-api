import streamlit as st
import json
import os
from dotenv import load_dotenv
from litellm import completion
from datetime import datetime  # Added import for timestamp
from streamlit_test_suite import test_suite_page
import time
import steer_api_client  # Importing the API client
from config import DEFAULT_NUM_CONTROL_DIMENSIONS, DEFAULT_NUM_SYNONYMS
from steer_templates import DEFAULT_PROMPT_LIST, SIMPLE_PROMPT_LIST, MODEL_LOCAL_SAVE_PATH

# Load environment variables
load_dotenv()


# Initialize session state for current_model if not present
if 'current_model' not in st.session_state:
    st.session_state['current_model'] = None

################################################
# Data Helpers 
################################################

def generate_positive_negative_examples(word):
    prompt = f"""Here is a word, please create an ordered list of {DEFAULT_NUM_SYNONYMS} SYNONYMS (similar to the word) and then {DEFAULT_NUM_SYNONYMS} ANTONYMS (the opposite of the word). Respond ONLY with a JSON object containing two keys: "positive_examples" and "negative_examples". Each key should map to a list of 5 examples. Respond with the following format:

    =====
    EXAMPLE INPUT:
    materialistic

    EXAMPLE RESPONSE:
    {{
        "positive_examples": ["materialistic", "consumerist", "acquisitive", "wealthy", "greedy", "possessive", "money-oriented", "worldly", "profit-driven", "commercialistic"],
        "negative_examples": ["minimalist", "austere", "spiritual", "altruistic", "ascetic", "frugal", "selfless", "anti-consumerist", "non-materialistic", "content"]
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


################################################
# Save and Display Models  
################################################

def select_model(model_id):
    """
    Callback function to set the selected model in session_state.

    Args:
        model_id (str): The ID of the model to select.
    """
    st.session_state['current_model'] = model_id

# Add this function to handle local file operations
def load_saved_models_from_file():
    # Create the model_data directory if it doesn't exist
    os.makedirs('model_data', exist_ok=True)
    
    # Load the models from the file in the model_data directory
    try:
        with open(MODEL_LOCAL_SAVE_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_models_to_file(models):
    # Create the model_data directory if it doesn't exist
    os.makedirs('model_data', exist_ok=True)
    
    # Save the models to a file in the model_data directory
    with open(MODEL_LOCAL_SAVE_PATH, 'w') as f:
        json.dump(models, f)


def display_saved_models():
    """
    Displays the saved models with SELECT buttons for selection
    and properly formatted Control Dimensions by fetching from local JSON.
    """
    saved_models = load_saved_models_from_file()


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
            with st.expander(f"### Model:   **{model['id']}**", expanded=is_selected): # Model Name 
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

################################################
# Main Steer Model Page 
################################################

# Do not delete the health check
def api_health_check():
    # Perform health check
    try:
        response = steer_api_client.health_check()
        st.success(f"Successfully connected to API: {response}")
    except Exception as e:
        st.error(f"Failed to connect to API: {str(e)}")

def steer_model_page():
    st.title("Steer Model")

    ################################################
    # Create Model 
    ################################################

    api_health_check()

    st.markdown("---")

    st.header("Create Steerable Model")

    # Initialize session state for dynamic control dimensions
    if 'num_control_dimensions' not in st.session_state:
        st.session_state.num_control_dimensions = DEFAULT_NUM_CONTROL_DIMENSIONS

    # Create column headers
    header_cols = st.columns([1, 1, 2])
    with header_cols[0]:
        st.write("**Control Word**")
        st.write("The type of behavior you want to steer.")
    with header_cols[2]:
        st.write("**Control Dimensions**")
        st.write('Click "Generate Examples" to generate a list of positive and negative examples for the control word, or enter your own list, formatted as JSON. ')

    placeholder_words = ["missionary", "savvy", "sophisticated"]
    
    # Create dynamic rows for control dimensions
    for i in range(st.session_state.num_control_dimensions):
        # if f'word_{i}' not in st.session_state:
            # st.session_state[f'word_{i}'] = 'empty'
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
                st.session_state.num_control_dimensions = DEFAULT_NUM_CONTROL_DIMENSIONS  # Default value
            st.session_state.num_control_dimensions += 1
            st.rerun()
    with col2:
        if st.button("➖ Remove Word"):
            if 'num_control_dimensions' not in st.session_state:
                st.session_state.num_control_dimensions = DEFAULT_NUM_CONTROL_DIMENSIONS  # Default value
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
            # Display success message after starting the training
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
                # Prepare the data payload for the API
                api_control_dimensions = {}
                for word, examples in control_dimensions.items():
                    api_control_dimensions[word] = {
                        "positive_examples": examples.get("positive_examples", []),
                        "negative_examples": examples.get("negative_examples", [])
                    }

                try:
                    # Call the API to create a steerable model
                    response = steer_api_client.create_steerable_model(
                        model_label=model_name,
                        control_dimensions=api_control_dimensions,
                        prompt_list=SIMPLE_PROMPT_LIST
                    )

                    model_id = response['id']

                    # Display success message without waiting for model to be ready
                    st.success(f"Started training model: {model_name} (ID: {model_id})")
                    st.session_state['current_model'] = model_id  # Set as current model

                except Exception as e:
                    st.error(f"Failed to create model: {str(e)}")
            else:
                st.warning("No valid control dimensions provided. Model not saved.")

    ################################################
    # Pull Completed Models
    ################################################

    st.markdown("---")
    st.header("Manage Models")

    if 'models_list' not in st.session_state:
        st.session_state['models_list'] = []

    if st.button("⬇️  Pull Completed Models"):
        try:
            # Retrieve the list of models from the API
            models_list = steer_api_client.list_steerable_models()
            st.session_state['models_list'] = models_list  # Store in session state
            st.success("Successfully retrieved the latest models.")
        except Exception as e:
            st.error(f"Failed to retrieve models: {str(e)}")

    # Load models from session state
    models_list = st.session_state['models_list']

    # Separate models into pending and ready
    pending_models = [model for model in models_list if model.get('status') == 'pending']
    ready_models = [model for model in models_list if model.get('status') == 'ready']

    ################################################
    # Display Pending Models
    ################################################

    if pending_models:
        st.markdown("### ⏳ Pending Models")
        for model in pending_models:
            with st.expander(f"Model: **{model['id']}** - **{model.get('label', 'N/A')}**"):
                st.write(f"**Status:** {model.get('status')}")
                st.write(f"**Created At:** {model.get('created_at')}")
    else:
        st.markdown("### ⏳ Pending Models")
        st.write("No pending models.")

    ################################################
    # Display Saved Models
    ################################################

    if ready_models:
        st.markdown("### ✅ Saved Models")
        for model in ready_models:
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
                with st.expander(f"Model: **{model['id']}** - **{model.get('label', 'N/A')}**", expanded=is_selected):
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
    else:
        st.markdown("### ✅ Saved Models")
        st.write("No saved models available.")

    ################################################
    # Prepare to Generate 
    ################################################

    # Third section: Generate
    st.markdown("---")
    st.markdown("### 💬 Generate")

    if st.session_state.get('current_model') is None:
        st.warning("Please select a model from the 'Saved Models' section above.")
    else:
        # Display the selected model ID
        selected_model_id = st.session_state['current_model']
        st.write(f"**Generating with Model ID:** `{selected_model_id}`")

        # Load the model details from the API
        try:
            selected_model = steer_api_client.get_steerable_model(selected_model_id)
            if not selected_model:
                st.error("Selected model not found in saved models.")
                return
        except Exception as e:
            st.error(f"Error loading model details: {str(e)}")
            return

        # Sliders for control dimensions
        control_dimensions = selected_model.get('control_dimensions', {})

        if control_dimensions:
            st.markdown("#### Adjust Control Dimensions")
            for word in control_dimensions.keys():
                col1, col2 = st.columns([1, 5])
                
                with col1:
                    number_value = st.number_input(
                        label=f"{word}",
                        min_value=-10,
                        max_value=10,
                        value=0,
                        step=1,
                        key=f"number_{word}"
                    )
                
                with col2:
                    slider_value = st.slider(
                        label=f"",
                        min_value=-10,
                        max_value=10,
                        value=0,
                        key=f"slider_{word}"
                    )
        else:
            st.info("This model has no control dimensions.")

        ################################################
        # Chat with Steered Model 
        ################################################
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Enter your message..."):
            # Collect control settings only when sending a message
            control_settings = {}
            for word in control_dimensions.keys():
                control_settings[word] = st.session_state[f"slider_{word}"]

            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response via the API
            try:
                response = steer_api_client.generate_completion(
                    model_id=selected_model_id,
                    prompt=prompt,
                    control_settings=control_settings,
                    settings={"max_new_tokens": 256}
                )
                # Parse the 'content' from the response
                full_response = response.get('content', "No content found in the response.")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                full_response = "An error occurred while generating the response."

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            
            # Rerun the app to update the chat display
            st.rerun()

        # Refresh Chat button
        if st.button("Refresh Chat"):
            st.session_state.chat_history = []
            st.rerun()

################################################
# Main 
################################################
def main():
    st.set_page_config(page_title="Steerable Models App", layout="wide")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Steer Model", "API Documentation", "Research Notebook", "Test Suite"])
    
    with tab1:
        steer_model_page()
    
    with tab2:
        st.write("API Documentation")
    
    with tab3:
        st.write("Research Notebook")
    
    with tab4:
        test_suite_page()

if __name__ == "__main__":
    main()
