import streamlit as st
import json
import os
from dotenv import load_dotenv
from litellm import completion
from datetime import datetime  # Added import for timestamp
from streamlit_test_suite import test_suite_page
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import client.steer_api_client as steer_api_client  # Importing the API client
from client.config import DEFAULT_NUM_CONTROL_DIMENSIONS, DEFAULT_NUM_SYNONYMS
from server.steer_templates import DEFAULT_PROMPT_LIST, SIMPLE_PROMPT_LIST, MODEL_LOCAL_SAVE_PATH
import nbformat
from nbconvert import MarkdownExporter

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
    Callback function to set the selected model in session_state and refresh the chat.

    Args:
        model_id (str): The ID of the model to select.
    """
    st.session_state['current_model'] = model_id
    st.session_state.chat_history = []
    st.session_state.waiting_for_response = False
    st.rerun()

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
        st.success(f"Successfully connected to API: {response['message']}")
    except Exception as e:
        st.error(f"Failed to connect to API: {str(e)}")

def on_value_change(word):
    # Determine which widget changed and update the shared value
    number_value_key = f"number_value_{word}"
    slider_value_key = f"slider_value_{word}"
    if st.session_state[slider_value_key] != st.session_state[f"value_{word}"]:
        st.session_state[f"value_{word}"] = st.session_state[slider_value_key]
    elif st.session_state[number_value_key] != st.session_state[f"value_{word}"]:
        st.session_state[f"value_{word}"] = st.session_state[number_value_key]
    

def steer_model_page():
    # Create two columns for the image and title
    col1, col2 = st.columns([1, 14])

    # Display the image in the first column
    with col1:
        st.image('client/images/icon_steer_square.jpg', width=100)  # Adjust width as needed

    # Display the title in the second column
    with col2:
        st.title("Steer API")

    api_health_check()

    ################################################
    # Create Model 
    ################################################

    st.markdown("---")

    st.header("Create Steering Vectors")

    # Initialize session state for dynamic control dimensions
    if 'num_control_dimensions' not in st.session_state:
        st.session_state.num_control_dimensions = DEFAULT_NUM_CONTROL_DIMENSIONS

    # Create column headers
    header_cols = st.columns([1, 1, 1, 1])
    with header_cols[0]:
        st.write("**Control Word**")
        st.write("The type of behavior you want to steer")
    with header_cols[1]:
        st.write("**Generate**")
        st.write("Auto-suggest examples")
    with header_cols[2]:
        st.write("**Positive Examples**")
        st.write("Each example on a new line")
    with header_cols[3]:
        st.write("**Negative Examples**")

    placeholder_words = ["missionary", "savvy", "sophisticated"]
    
    # Create dynamic rows for control dimensions
    for i in range(st.session_state.num_control_dimensions):
        if f'positive_examples_{i}' not in st.session_state:
            st.session_state[f'positive_examples_{i}'] = "example1\nexample2"
        if f'negative_examples_{i}' not in st.session_state:
            st.session_state[f'negative_examples_{i}'] = "example3\nexample4"

        row_cols = st.columns([1, 1, 1, 1])
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
                                positive = "\n".join(result.get("positive_examples", []))
                                negative = "\n".join(result.get("negative_examples", []))
                                st.session_state[f'positive_examples_{i}'] = positive
                                st.session_state[f'negative_examples_{i}'] = negative
                                st.success(f"Examples generated for '{st.session_state[f'word_{i}']}'.")
                            else:
                                st.error("Unexpected result format from generate_positive_negative_examples")
                        except Exception as e:
                            st.error(f"Failed to parse generated examples: {str(e)}")
        with row_cols[2]:
            st.text_area(
                label=f"Positive Examples {i+1}",
                key=f"positive_examples_{i}",
                height=150,
                label_visibility="collapsed",
                value=st.session_state[f'positive_examples_{i}']
            )
        with row_cols[3]:
            st.text_area(
                label=f"Negative Examples {i+1}",
                key=f"negative_examples_{i}",
                height=150,
                label_visibility="collapsed",
                value=st.session_state[f'negative_examples_{i}']
            )

    # Button to add new control dimension
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
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
    st.markdown("#### Name this vector set")
    
    # Automatically populate the model name with control dimensions
    if 'model_to_create_name' not in st.session_state:
        st.session_state['model_to_create_name'] = ''

    control_words = [
        st.session_state.get(f'word_{i}', '').strip() 
        for i in range(st.session_state.num_control_dimensions)
    ]
    # Filter out any empty strings
    control_words = [word for word in control_words if word]
    # Join with hyphens
    default_model_name = '-'.join(control_words)
    st.session_state['model_name'] = default_model_name
    
    model_name = st.text_input(
        "", 
        key="model_name", 
        label_visibility='hidden',
        value=st.session_state['model_to_create_name']
    )

    # "Create Model" button
    if st.button("Create Model"):
        if not model_name:
            st.error("Please enter a model name.")
        else:
            # Prepare control dimensions
            control_dimensions = {}
            for i in range(st.session_state.num_control_dimensions):
                control_word = st.session_state.get(f'word_{i}', '').strip()
                positive_examples = st.session_state.get(f'positive_examples_{i}', '').strip()
                negative_examples = st.session_state.get(f'negative_examples_{i}', '').strip()

                if control_word and positive_examples and negative_examples:
                    try:
                        control_dimensions[control_word] = {
                            "positive_examples": positive_examples.split("\n"),
                            "negative_examples": negative_examples.split("\n")
                        }
                    except Exception as e:
                        st.error(f"Error parsing control dimensions: {str(e)}")

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
                st.warning("No valid control dimensions provided. Vectors not saved.")

    ################################################
    # Pull Completed Models
    ################################################

    st.markdown("---")
    st.header("Manage Steering Vectors")

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
        st.markdown("### ⏳ Pending Vectors")
        for model in pending_models:
            with st.expander(f"Model: **{model['id']}** - **{model.get('label', 'N/A')}**"):
                st.write(f"**Status:** {model.get('status')}")
                st.write(f"**Created At:** {model.get('created_at')}")
    else:
        st.markdown("### ⏳ Pending Vectors")
        st.write("No pending models.")

    ################################################
    # Display Saved Models
    ################################################

    if ready_models:
        st.markdown("### ☀️ Saved Vectors")
        for model in ready_models:
            cols = st.columns([0.15, 1])  # First column ~15% width for the SELECT button

            with cols[0]:
                # Determine if this model is currently selected
                is_selected = st.session_state.get('current_model') == model['id']

                # Define button label based on selection
                button_label = "✅ Current Vectors" if is_selected else "🔲 Select vectors"

                # The 'disabled' parameter visually indicates selection by disabling the button
                st.button(
                    button_label,
                    key=f"select_{model['id']}",
                    on_click=select_model,
                    args=(model['id'],),
                    disabled=is_selected,  # Disable if selected
                )

            with cols[1]:
                with st.expander(f"Vectors: **{model['id']}** - **{model.get('label', 'N/A')}**", expanded=is_selected):
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
                        st.markdown("**No control dimensions provided for this vector set.**")

                    # Highlight if this model is currently selected
                    if is_selected:
                        st.success("**This vector set is currently selected.**")
    else:
        st.markdown("### ☀️ Saved Vectors")
        st.write("No saved vectors available.")

    ################################################
    # Prepare to Generate 
    ################################################

    # Third section: Generate
    st.markdown("---")
    st.markdown("## 💬 Generate")

    if st.session_state.get('current_model') is None:
        st.warning("Please select a vector set from the 'Saved Vectors' section above.")
    else:
        # Display the selected model ID
        selected_model_id = st.session_state['current_model']
        st.write(f"**Generating with Model ID:** `{selected_model_id}`")
        # Load the model details from the API
        try:
            selected_model = steer_api_client.get_steerable_model(selected_model_id)
            if not selected_model:
                st.error("Selected vector not found in saved vectors.")
                return
        except Exception as e:
            st.error(f"Error loading vector details: {str(e)}")
            return

        # Sliders for control dimensions
        control_dimensions = selected_model.get('control_dimensions', {})
        print('control_dimensions: ', control_dimensions)

    ################################################
    # Added Quick Test Section
    ################################################
    st.markdown("#### Quick Test")

    # Initialize session state for Quick Test if not present
    if 'quick_test_dimension' not in st.session_state:
        if control_dimensions:
            st.session_state['quick_test_dimension'] = list(control_dimensions.keys())[0]
        else:
            st.session_state['quick_test_dimension'] = None

    # Radio buttons to select a control dimension
    if control_dimensions:
        selected_dimension = st.radio(
            "Select Control Dimension:",
            options=list(control_dimensions.keys()),
            index=0,
            key="quick_test_dimension"
        )
    else:
        st.warning("No control dimensions available for Quick Test.")
        selected_dimension = None

    # Text box for user input
    quick_test_input = st.text_input("Enter your prompt for Quick Test:", key="quick_test_input")

    # Generate button with spinner
    if st.button("Run Quick Test"):
        if not selected_dimension:
            st.error("No control dimension selected.")
        elif not quick_test_input:
            st.error("Please enter a prompt.")
        else:
            with st.spinner("Generating Quick Test results..."):
                # Prepare control settings with specified dimension values
                control_settings_neg = {selected_dimension: -1.5}
                control_settings_zero = {selected_dimension: 0.0}
                control_settings_pos = {selected_dimension: 1.5}

                try:
                    # Make three completion requests
                    response_zero = steer_api_client.generate_completion(
                        model_id=selected_model_id,
                        prompt=quick_test_input,
                        control_settings=control_settings_zero,
                        settings={"max_new_tokens": 256}
                    )
                    response_neg = steer_api_client.generate_completion(
                        model_id=selected_model_id,
                        prompt=quick_test_input,
                        control_settings=control_settings_neg,
                        settings={"max_new_tokens": 256}
                    )
                    response_pos = steer_api_client.generate_completion(
                        model_id=selected_model_id,
                        prompt=quick_test_input,
                        control_settings=control_settings_pos,
                        settings={"max_new_tokens": 256}
                    )

                    # Extract content from responses
                    def extract_content(resp):
                        if isinstance(resp, dict) and 'content' in resp:
                            return resp['content']
                        elif isinstance(resp, str):
                            return resp
                        else:
                            return "Invalid response format."

                    baseline = extract_content(response_zero)
                    neg_result = extract_content(response_neg)
                    pos_result = extract_content(response_pos)

                    # Display the results
                    st.markdown("**Results:**")
                    st.markdown(f"**Baseline (0):** {baseline}")
                    st.markdown(f"**-- (-1.5):** {neg_result}")
                    st.markdown(f"**++ (+1.5):** {pos_result}")

                except Exception as e:
                    st.error(f"Error during Quick Test generation: {str(e)}")

    st.markdown('---')
        
        
        ################################################
        # Chat with Steered Model 
        ################################################


            ################################################
            # Sliders for control dimensions

    if control_dimensions:
        st.markdown("#### Custom Chat")
        for word in control_dimensions.keys():
            # Initialize session state for this word if not already done
            if f"value_{word}" not in st.session_state:
                st.session_state[f"value_{word}"] = 0.0

            col1, col2 = st.columns([1, 5])

            with col1:
                number_value_key = f"number_value_{word}"
                st.number_input(
                    label=f"{word}",
                    min_value=-5.0,
                    max_value=5.0,
                    value=st.session_state[f"value_{word}"],
                    step=0.5,
                    key=number_value_key,
                    on_change=on_value_change,
                    args=(word,),  # Pass 'word' as an argument to the callback
                )

            with col2:
                slider_value_key = f"slider_value_{word}"
                st.slider(
                    label="",
                    min_value=-5.0,
                    max_value=5.0,
                    value=st.session_state[f"value_{word}"],
                    step=0.5,
                    key=slider_value_key,
                    on_change=on_value_change,
                    args=(word,),  # Pass 'word' as an argument to the callback
                )

    # Initialize chat history and other states if they don't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'waiting_for_response' not in st.session_state:
        st.session_state.waiting_for_response = False
    if 'control_settings' not in st.session_state:
        st.session_state.control_settings = {}
    # if 'needs_rerun' not in st.session_state:
    #     st.session_state.needs_rerun = False

    # Create a container for the chat history
    chat_container = st.container()

    # Chat input
    user_input = st.chat_input("Enter your message...")

    if user_input:
        # Collect control settings when sending a message
        control_settings = {}
        for word in control_dimensions.keys():
            control_settings[word] = st.session_state[f"value_{word}"]
        st.session_state.control_settings = control_settings
        st.session_state.waiting_for_response = True
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.needs_rerun = True

    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Display "..." message if waiting for response
        if st.session_state.waiting_for_response:
            with st.chat_message("assistant"):
                st.markdown("...")

    # Process the API response
    if st.session_state.waiting_for_response:
        try:
            response = steer_api_client.generate_completion(
                model_id=selected_model_id,
                prompt=st.session_state.chat_history[-1]["content"],
                control_settings=st.session_state.control_settings,
                settings={"max_new_tokens": 256}
            )
            
            # Parse the response to get only the content
            if isinstance(response, dict) and 'content' in response:
                full_response = response['content']
            elif isinstance(response, str):
                full_response = response
            else:
                raise ValueError("Unexpected response format from API")
            
            # Format control settings with smaller font using Markdown
            control_settings_str = ", ".join([f"{k}: {v}" for k, v in st.session_state.control_settings.items()])
            control_settings_md = f"{control_settings_str}"
            
            # Combine control settings and response
            full_response_with_settings = f"""
[{control_settings_md}]

{full_response}
"""
            
        except Exception as e:
            full_response_with_settings = f"Error generating response: {str(e)}"
            st.error(full_response_with_settings)

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response_with_settings})
        st.session_state.waiting_for_response = False
        st.rerun()  # Rerun to display the new message

    st.markdown('---')
    if st.button("Refresh Chat"):
        st.session_state.chat_history = []
        st.session_state.waiting_for_response = False
        st.rerun()  # Rerun the app to refresh the chat

    st.markdown('---')
    if st.button("🤠", key="balloons_button"):
        st.balloons()
################################################
# API Docs page 
################################################

def load_api_docs():
    """Load and return the content of api_docs.md from the parent directory."""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    api_docs_path = os.path.join(parent_dir, 'api_docs.md')
    try:
        with open(api_docs_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "API documentation file not found."
    except Exception as e:
        return f"Error loading API documentation: {str(e)}"
    

################################################
# Research Notebook page 
################################################

def load_jupyter_notebook():
    """Load and return the content of the Jupyter notebook as markdown."""
    notebook_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'quickstart', 'samples.ipynb')
    try:
        with open(notebook_path, 'r', encoding='utf-8') as file:
            notebook = nbformat.read(file, as_version=4)
        
        # Convert notebook to markdown
        exporter = MarkdownExporter()
        markdown, _ = exporter.from_notebook_node(notebook)
        return markdown
    except FileNotFoundError:
        return "Jupyter notebook file not found."
    except Exception as e:
        return f"Error loading Jupyter notebook: {str(e)}"

################################################
# Main 
################################################
def main():
    st.set_page_config(page_title="Steerable Models App", layout="wide")

    # Create tabs
    tab1, tab2, tab3= st.tabs(["Steer API", "API Documentation", "Examples Notebook"])
    
    with tab1:
        steer_model_page()
    
    with tab2:
        api_docs_content = load_api_docs()
        REMOTE_URL = os.getenv('REMOTE_URL')
        st.markdown(f"#### API Server URL: `{REMOTE_URL}`")
        st.markdown(api_docs_content)
    
    with tab3:
        notebook_content = load_jupyter_notebook()
        st.markdown(notebook_content)
    
    # with tab4:
    #     test_suite_page()

if __name__ == "__main__":
    main()

