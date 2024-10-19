import streamlit as st
import json
import os

def test_suite_page():
    st.title("Test Suite")

    # Access the current model
    current_model = st.session_state.get('current_model')
    
    if current_model:
        st.write(f"Current model: {current_model}")
        
        # Load the model details
        if os.path.exists("models.json"):
            with open("models.json", "r") as f:
                models_data = json.load(f)
                selected_model = next((model for model in models_data if model['id'] == current_model), None)
        
            if selected_model:
                control_dimensions = selected_model.get('control_dimensions', {})
                
                # Access slider values
                for word in control_dimensions.keys():
                    slider_value = st.session_state.get(f"slider_value_{word}", 0)
                    st.write(f"Slider value for {word}: {slider_value}")

                # Rest of your test suite code...
                # List of prompts
                default_prompts = """Write a story about a brave knight.

                Describe a futuristic city.

                Explain the process of photosynthesis.

                Write a recipe for chocolate chip cookies.

                Discuss the impact of social media on society."""

                test_prompts = st.text_area("List of Test Prompts (separated by double-lines)", value=default_prompts, height=200, key="test_prompts")

                # Testing prompt
                default_testing_prompt = f"Please act like this: {json.dumps(selected_model['control_dimensions'])}"
                testing_prompt = st.text_area("Control Prompt", value=default_testing_prompt, height=100, key="testing_prompt")

                # Process and display results for each test prompt
                prompts = [p.strip() for p in test_prompts.split('\n\n') if p.strip()]

                if st.button("Generate Test Suite"):
                    st.info("Test suite generation initiated. This may take a moment...")
                    # Here you would typically call a function to generate the test suite
                    st.success("Test suite generated successfully!")

                st.markdown("---")
                st.markdown(f"\n#### Responses:")
                for prompt in prompts:
                    st.markdown(f"**{prompt}**")

                    col1, col2, col3, col4 = st.columns(4)

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
            else:
                st.error("Selected model not found in saved models.")
        else:
            st.error("models.json file not found.")
    else:
        st.warning("No model selected. Please select a model in the Steer Model tab.")