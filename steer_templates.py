BASE_MODEL_NAME = "aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored"

user_tag = "<|start_header_id|>user<|end_header_id|>You: "
asst_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>Assistant:"

DEFAULT_TEMPLATE = "{user_tag} Act as if you're extremely {persona}. {asst_tag}{suffix}"

DEFAULT_SUFFIX_LIST = [
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

SIMPLE_SUFFIX_LIST = [
    "", "That game", "I can see", "Hmm, this", "I can relate to", "Who is"]