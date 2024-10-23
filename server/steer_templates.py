DEFAULT_TEMPLATE = "{user_tag} Respond as though you're an extremely {persona} person. {asst_tag}{prompt}"

user_tag = "<|start_header_id|>user<|end_header_id|>You: "
asst_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>Assistant:"

prompt_filepaths = {
    "emotions": "data_full/emotions_small.json",
    "facts": "data_full/facts_small.json",
    "all_truncated_outputs": "data/all_truncated_outputs.json"
}

BASE_MODEL_NAME = "aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored"

MODEL_LOCAL_SAVE_PATH='model_data/saved_models.json'

DEFAULT_PROMPT_LIST = [
    "", "That game", "I can see", "Hmm, this", "I can relate to", "Who is",
    "I understand the", "Ugh,", "What the hell was", "Hey, did anyone", "Although",
    "Thank you for choosing", "What are you", "Oh w", "How dare you open",
    "It was my pleasure", "I'm hon", "I appreciate that you", "Are you k",
    "Whoever left this", "It's always", "Ew,", "Hey, I l", "Hello? Is someone",
    "I understand that", "That poem", "Aww, poor", "Hey, it", "Alright, who",
    "I didn't", "Well, life", "The document", "Oh no, this", "I'm concerned",
    "Hello, this is", "This art", "Hmm, this drink", "Hi there!", "It seems",
    "Is", "Good", "I can't", "Ex", "Who are", "I can see that", "Wow,",
    "Today is a", "Hey friend", "Sometimes friends"
]

SIMPLE_PROMPT_LIST = [
    "", "That game", "I can see", "Hmm, this", "I can relate to", "Who is"]