from config import OPENAI_API_KEY
import openai

# Function to get completions from OpenAI chat model
def get_chat_completion(model, system_msg, query):
    """Get completions from OpenAI chat model."""
    return openai.Client(api_key=OPENAI_API_KEY).chat.completions.create(
        model=model,
        messages=[
            {
                'role': "system", 
                "content": system_msg
            },
            {
                "role": "user", 
                "content": query
            }
        ],
        temperature=0.0,
        max_tokens=150,
    )