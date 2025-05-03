
import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv
import base64

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")




# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# If using Anthropic's Python library for Claude (hypothetical usage):
#   pip install anthropic
#   import anthropic
#   anthropic.Client(ANTHROPIC_API_KEY)

def generate_llm_response(prompt, provider="openai", model="gpt-4o", temperature=0.7):
    """
    Generates a response from various LLM providers (OpenAI, Hugging Face, Claude, Google Gemini).
    
    :param prompt: The prompt or query string.
    :param provider: Which LLM provider to use ('openai', 'huggingface', 'claude', 'gemini').
    :param model: Model name (e.g., 'gpt-4', 'gpt-4o', 'claude-v1', 'google-gemini', etc.).
    :param temperature: Sampling temperature (if applicable).
    :return: The text response from the LLM, or an error string if something fails.
    """
    try:
   
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
        return response.choices[0].message.content   
    
    except Exception as e:
        return f"LLM Error: {str(e)}"
    
    
def generate_image_description(image_path, prompt,provider="openai", model="gpt-4o-mini",temperature=0.7):
    """
    Generates an image description using OpenAI's API.
    Since OpenAI's API does not accept image binary directly for captioning,
    the image is assumed to be available at a public URL.
    
    :param image_path: Path to the input image.
    :param provider: LLM provider, default 'openai'.
    :param model: LLM model name.
    :param temperature: Sampling temperature.
    :return: A generated caption describing the image.
    """
    client = OpenAI(api_key=OPENAI_API_KEY) 
    
    image_path = image_path
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
                                            model=model,
                                            messages=[
                                                                    {
                                                                        "role": "user",
                                                                        "content": [
                                                                            {
                                                                                "type": "text",
                                                                                "text": prompt,
                                                                            },
                                                                            {
                                                                                "type": "image_url",
                                                                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                                                            },
                                                                        ],
                                                                    }
                                                                ],
                                                            )



    return response.choices[0].message.content



def generate_llm_json(prompt,event,provider="openai", model="gpt-4o-2024-08-06",temperature=0.7):
    try:
        if provider.lower() == "openai":
            client = OpenAI(api_key=OPENAI_API_KEY)
            completion = client.beta.chat.completions.parse(
            model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            response_format=event,
            )
            return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return f"LLM Error: {str(e)}"
    