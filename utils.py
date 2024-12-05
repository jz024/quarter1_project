import openai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("SAMBANOVA_API_KEY")
client = openai.OpenAI(base_url="https://api.sambanova.ai/v1", api_key=api_key)

def count_tokens(text, chars_per_token=4):
    return len(text) // chars_per_token

def estimate_cost(input_tokens, output_tokens, input_price, output_price):
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    return input_cost + output_cost
