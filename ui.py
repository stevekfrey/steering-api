import streamlit as st
from streamlit_option_menu import option_menu
import json
import os
from dotenv import load_dotenv
from litellm import completion

# Load environment variables
load_dotenv()

def generate_synonyms_antonyms(word):
    prompt = f"""Here is a word, please create an ordered list of 5 SYNONYMs (similar to the word) and then 5 ANTONYMS (the opposite of the word). Respond ONLY with a list of lists, starting with the '[' and ending with the ']'. Respond with the following format:

=====
EXAMPLE INPUT:
materialistic

EXAMPLE RESPONSE:
[
    ["materialistic", "consumerist", "acquisitive", "wealthy", "greedy"],
    ["minimalist", "austere", "spiritual", "altruistic", "ascetic"]
]
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
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating synonyms and antonyms: {str(e)}")
        return None



def parse_syn_ant(syn_ant_str):
    """
    Parses a string containing synonyms and antonyms separated by newline characters.

    Args:
        syn_ant_str (str): The string containing synonyms and antonyms.

    Returns:
        tuple: A tuple containing two strings: synonyms and antonyms.
    """
    synonyms = ""
    antonyms = ""

    lines = syn_ant_str.strip().split('\n')
    for line in lines:
        if line.lower().startswith('synonyms:'):
            synonyms = line.replace('synonyms:', '').strip()
        elif line.lower().startswith('antonyms:'):
            antonyms = line.replace('antonyms:', '').strip()

    return synonyms, antonyms


def display_saved_models(saved_models):
    """
    Displays the saved models with properly formatted Synonyms and Antonyms.

    Args:
        saved_models (list of dict): A list where each dict contains model details,
                                    including a 'syn_ant' string with synonyms and antonyms separated by newline characters.
    """
    st.header("Saved Models")

    for model in saved_models:
        with st.expander(model['name']):
            syn_ant_str = model.get('syn_ant', '')
            synonyms, antonyms = parse_syn_ant(syn_ant_str)
            
            if synonyms:
                st.markdown(f"**Synonyms:** {synonyms}")
            else:
                st.markdown("**Synonyms:** _None provided_")
            
            if antonyms:
                st.markdown(f"**Antonyms:** {antonyms}")
            else:
                st.markdown("**Antonyms:** _None provided_")
                
def steer_model_page():
    st.title("Steer Model")

    st.header("Create Steerable Model")

    # Initialize session state for text inputs and text areas
    for i in range(3):
        if f'word_{i}' not in st.session_state:
            st.session_state[f'word_{i}'] = ''
        if f'syn_ant_{i}' not in st.session_state:
            st.session_state[f'syn_ant_{i}'] = ''

    # Create column headers
    header_cols = st.columns([1, 1, 2])
    with header_cols[0]:
        st.write("**Control Word**")
        st.write("The type of behavior you want to steer.")
    # with header_cols[1]:
        # st.write("**Generate Examples**")
    with header_cols[2]:
        st.write("**Synonyms/Antonyms**")
        st.write("Click 'Generate Examples' to generate a list of synonyms and antonyms for the control word, or enter your own list.")

    # Create 3 rows with text input, generate button, and synonyms/antonyms text area
    for i in range(3):
        row_cols = st.columns([1, 1, 2])
        with row_cols[0]:
            st.text_input(
                label=f"control_word_{i}",
                key=f"word_{i}",
                label_visibility='collapsed'
            )
        with row_cols[1]:
            # Use a unique button key for each button
            if st.button("Generate examples", key=f"generate_btn_{i}"):
                if st.session_state[f'word_{i}']:
                    result = generate_synonyms_antonyms(st.session_state[f'word_{i}'])
                    if result:
                        st.session_state[f'syn_ant_{i}'] = result
        with row_cols[2]:
            placeholder_text = (
                'Enter a list of synonyms and antonyms like below, or click "generate"\n'
                '[synonyms: "", "", "", "", "", antonyms: "", "", "", "", ""]'
            )
            st.text_area(
                label=f"word_examples_{i}",
                placeholder=placeholder_text,
                key=f"syn_ant_{i}",
                height=200,
                label_visibility='collapsed'
                # Removed the `value` parameter
            )

    # Text box to input the model name
    model_name = st.text_input("Name this model", key="model_name")

    # "Create Model" button
    if st.button("Create Model"):
        if not model_name:
            st.error("Please enter a model name.")
        else:
            # Placeholder for API call (to be implemented later)
            st.write("Model creation functionality will be implemented here.")

            # Simulate an API response
            api_response = {
                "id": f"model_{len([k for k in st.session_state.keys() if 'word_' in k])}",
                "name": model_name,
                "words": [st.session_state[f'word_{i}'] for i in range(3)],
                "synonyms_antonyms": [st.session_state[f'syn_ant_{i}'] for i in range(3)]
            }

            # Save the response locally in a JSON file
            # Load existing data
            models_data = []
            if os.path.exists("models.json"):
                with open("models.json", "r") as f:
                    models_data = json.load(f)

            # Append new model
            models_data.append(api_response)

            # Save back to the file
            with open("models.json", "w") as f:
                json.dump(models_data, f, indent=4)

            st.success("Model created and saved locally.")

    # Display saved models
    st.subheader("Saved Models")
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


    
        # Sample data
    saved_models = [
        {
            'name': 'Model A',
            'syn_ant': 'synonyms: happy, joyful, elated\nantonyms: sad, miserable, gloomy'
        },
        {
            'name': 'Model B',
            'syn_ant': 'synonyms: quick, swift, rapid\nantonyms: slow, sluggish, lethargic'
        },
        {
            'name': 'Model C',
            'syn_ant': 'synonyms: bright, luminous\nantonyms: dim, dark'
        }
    ]

    display_saved_models(saved_models)

def main():
    st.set_page_config(page_title="Steerable Models App")
    steer_model_page()

if __name__ == "__main__":
    main()