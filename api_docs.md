# Steering API Documentation

## Introduction

[GitHub](https://github.com/stephenkfrey/steering-api)

[Steer.chat](https://steer.chat)


Create, manage, and use steerable language models with the Steering API. These models can be controlled along various dimensions to generate text with specific traits. 


## Authentication

All API requests require authentication using a bearer token. Include the token in the `Authorization` header of your requests:

```
Authorization: Bearer YOUR_API_KEY
```

## API Endpoints

### Steering Vectors

#### CREATE Steering Vectors

Creates a new set of steering vectors with specified control dimensions.

```http
POST /steerable-model
```

**Request Body:**

```json
{
  "model_label": "string",
  "control_dimensions": {
    "trait_name": {
      "positive_examples": ["example1","example2","example3"]
      "negative_examples": ["example1","example2","example3"]
    }
  },
  "prompt_list": ["string", "string", "string"]
}
```

**Response:**

```json
{
  "id": "string",
  "object": "steerable_model",
  "created_at": "string",
  "model": "string",
  "control_dimensions": {
    "trait_name": {
      "positive_examples": ["example1","example2","example3"]
      "negative_examples": ["example1","example2","example3"]
    }
  },
  "status": "string"
}
```

#### LIST Steerable Vectors

Retrieves a list of available sets of steering vectors. 

```http
GET /steerable-model
```

**Query Parameters:**

- `limit` (optional): Number of models to return (default: 10)
- `offset` (optional): Number of models to skip (default: 0)

**Response:**

```json
{
  "data": [
    {
      "id": "string",
      "object": "steerable_model",
      "created_at": "string",
      "model": "string",
      "control_dimensions": {
	    "trait_name": {
	      "positive_examples": ["example1","example2","example3"]
	      "negative_examples": ["example1","example2","example3"]
	    }
	  },
      "status": "string"
    }
  ]
}
```

#### GET Steerable Vectors

Retrieves details of a specific set of steering vectors. 

```http
GET /steerable-model/{model_id}
```

**Response:**

```json
{
  "id": "string",
  "object": "steerable_model",
  "created_at": "string",
  "model": "string",
  "control_dimensions": {
    "trait_name": {
      "positive_examples": ["example1","example2","example3"]
      "negative_examples": ["example1","example2","example3"]
    }
  },
  "status": "string"
}
```

#### DELETE a Set of Steering Vectors

Deletes a specific steerable model.

```http
DELETE /steerable-model/{model_id}
```

**Response:**

```json
{
  "id": "string",
  "object": "steerable_model",
  "deleted": true
}
```


### Completions

#### Generate Completion

Generates text completion using a set of steering vectors. 

```http
POST /completions
```

**Request Body:**

```json
{
  "model": "string",
  "prompt": "string",
  "control_settings": {
    "trait_name": 0.5
  },
  "settings": {
    "do_sample": false,
    "max_new_tokens": 256,
    "repetition_penalty": 1.1
  }
}
```

**Response:**

```json
{
  "model_id": "string",
  "object": "text_completion",
  "created": "string",
  "model": "string",
  "content": "string"
}
```

## Error Handling

The API uses standard HTTP response codes to indicate the success or failure of requests. Codes in the 2xx range indicate success, codes in the 4xx range indicate an error that resulted from the provided information (e.g., a required parameter was missing), and codes in the 5xx range indicate an error with our servers.


## Rate Limiting

The API implements rate limiting. If you exceed the rate limit, you will receive a 429 Too Many Requests response. The response will include a Retry-After header indicating how long to wait before making another request.


## Examples

### Create a set of Steering Vectors 

```python
import requests

url = REMOTE_URL + "/steerable-model"
headers = {
    "Authorization": f"Bearer {API_AUTH_TOKEN}",
    "Content-Type": "application/json"
}
data = {
    "model_label": "positive_sentiment",
    "control_dimensions": {
        "positivity": {
            "positive_examples": ["This is great!", "I love it!"],
            "negative_examples": ["This is terrible.", "I hate it."]
        }
    },
    "prompt_list": ["Write a product review:", "Describe your day:"]
}

response = requests.post(url, headers=headers, json=data)

print(f"Status Code: {response.status_code}")
print("Response Content:")
print(response.text)
model_id = response.json()['id']
print (f"Model ID: {model_id}")

```

### Generating a Completion

```python
import requests

url = f"{REMOTE_URL}/completions"
headers = {
    "Authorization": f"Bearer {API_AUTH_TOKEN}",  # Fixed string formatting
    "Content-Type": "application/json"
}
data = {
    "model": model_id, 
    "prompt": "Write a product review:",
    "control_settings": {
        "positivity": 0.8
    },
    "settings": {
        "max_new_tokens": 100
    }
}

response = requests.post(url, headers=headers, json=data)
message = response.json()['content']
print (message)

```

## Best Practices

1. Experiment with different prompts and control settings to achieve the best results for your use case.
2. Anecdotally, the best results come when steering vector multipliers are set between -2.0 and 2.0. Larger steering multipliers lead to unpredictable and unreliable model performance. 


## Feedback and Support

If you have any feedback or issues, DM me on [Twitter](www.x.com/stevewattsfrey).