# Steerable LLM API


### Notes on Steering Vectors 

By adding or subtracting control vectors to the model's internal activations, we can steer the output toward desired behavior. It's kind of like doing neuroscience on a brain to see what desired brain states look like, then implanting a high-density electical array in the brain to stimulate it to mimic the desired neural activation pattern. 

1. For each trait, we create and save a control vector
2. To steer the model, we scale these control vectors according to how much we want that trait to influence the output. This scales the control vector according to how much influence we want that trait to have
3. We add these vectors together to get a single control vector
4. We apply this control vector to the model


### API Notes 

If the trait isn't specified in control_settings, it defaults to 0.0

### Misc implementation notes 

####Formatting 
Since this is a completions model, we manually insert user and asst tags to structure the output: 
```
DEFAULT_TEMPLATE = "{user_tag} Act as if you're extremely {persona}. {asst_tag}{prompt}"
```


# Why a Steering API? (Business Context) 

Advantages of steering techniques over prompting and fine-tuning: 

1. Can offer more fine-grained control than a prompt, by adjusting the steering multiplier 
	- **prompt:** “slightly cheerful vs very cheerful vs very very very cheerful” 
	- **steering:** “cheerful = 1.5x"
2. Potentially more resistant to jailbreaking 
	- Prompts and jailbreaks just add tokens to distract from previous tokens; but steering can be applied to *every* token 
3. Potentially lower data requirements, relative to fine-tuning -> easier for clients and developers to use

### Use cases 

- Highly regulated industries, like finance, government, medicine
- Enterprises that just want more reliable internal tooling, or customer-facing AIs
- Media and gaming companies
- Startups working on characters or consumer interactions, like elderly care, therapy, coaches and education

# API Documentation

See `api_docs.md` 


# Research Background 
This is based on the research on Activation Engineering in [Representation Engineering: A Top-Down Approach to AI Transparency
](https://arxiv.org/abs/2310.01405) and [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/html/2312.06681v2). It uses the [repeng](https://github.com/vgel/repeng/tree/main) library to create Control Vectors.


 


# Future development

- Compare with guardrail tools like LlamaGuard. Where are there relative strengths in performance, cost and inference speed?
- How to make it easier to generate the appropriate data
- Make it easier to to run standard test suites. Eg run it on a subset of MMLU, as done in Section 7 of [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/html/2312.06681v2).
- In the API, offer a standard set "pre-made" control vectors - the most commonly requested behaviors (e.g. "responsible" "only for coding") 

# Future research

- How to limit the control dimensions, to prevent reverse-engineering of the model weights
- Given the parameter size of a model, how many test prompts or inputs are needed to derive the “full circuit” for a behavior? Do you pass in every conceivable sentence to squeeze the last bits of signal from model activations? What if there are faint sub-circuits that hide during testing but emerge from out-of-distribution inputs (or creative jailbreaking) in the wild? 
- Multi-agent systems with custom behaviors (e.g. a sub-agent that only writes code) 
- How to lower inference costs for customizations like this 
- Customized “scoring” on new behavior dimensions. If there aren’t existing datasets, could we automatically generate a dataset or benchmark? 
    - A naive version of this automated grading system could pass responses to Sonnet / GPT-4 with the following prompt:
    
    ```
    Please score the answer on a scale of 0 to 10, where the score reflects the extent to which the answer demonstrates the descriptor word. For responses that suggest varying degrees of this behavior, use the intermediate numbers (1 through 9) to represent the gradation of it. Respond only with a number, nothing else.
    
    Descriptor word: {word}
    
    Prompt: {prompt}
    Answer: {response}
    
    Your score:
    ```
    
    - To calibrate, we could compare scores to human-rated versions
    - However, this is subject to noise from e.g. variations in the type of model that is used to evaluate (Sonnet, GPT-4, etc)