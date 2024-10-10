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
DEFAULT_TEMPLATE = "{user_tag} Act as if you're extremely {persona}. {asst_tag}{suffix}"
```

####Control Vectors
Start with a zero vector: starting from zero ensures that if control_settings is empty or if all its weights are zero, then vector_mix will also correctly be set to 0 (ie no influence applied).

####Deployment
`gunicorn app:app --workers 2 --threads 1 --bind 0.0.0.0:5000`

