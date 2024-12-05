import os
import random
import time
import openai
from dotenv import load_dotenv
import json

with open('config.json') as config_file:
    config = json.load(config_file)

load_dotenv()
api_key = os.getenv("SAMBANOVA_API_KEY")

client = openai.OpenAI(
    base_url="https://api.sambanova.ai/v1",
    api_key=api_key
)

def count_tokens(text, chars_per_token=4):
    char_count = len(text)
    return int(char_count / chars_per_token)

def load_dataset(dataset_path, num_samples=10):
    category_samples = {}
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            all_files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
            selected_files = random.sample(all_files, min(num_samples, len(all_files)))
            category_samples[category] = []
            for filename in selected_files:
                file_path = os.path.join(category_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        category_samples[category].append(file.read())
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='latin-1') as file:
                        category_samples[category].append(file.read())
    return category_samples

def estimate_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1_000_000) * config["input_price_per_million"]
    output_cost = (output_tokens / 1_000_000) * config["output_price_per_million"]
    return input_cost + output_cost

def get_category_list(dataset_path):
    return [category for category in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, category))]

def direct_prompting_testing(data, categories, model=config["model_name"], token_budget=config["token_budget"], delay=config["delay"]):
    correct_predictions = 0
    total_cost = 0
    total_samples = 0
    category_list = ", ".join(categories)
    
    for category, samples in data.items():
        for text in samples:
            prompt_text = (f"{text}\n\n"
                           f"Classify this news article into one of the following categories: "
                           f"{category_list}. Your answer should be exactly one of these categories.")
            tokens = count_tokens(prompt_text)
            if tokens > token_budget:
                print(f"Skipping sample due to exceeding token budget: {tokens} tokens")
                continue
            
            total_samples += 1

            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                stream=True,
                temperature=0.1,
                top_p=0.1
            )

            response = ""
            for chunk in completion:
                response += chunk.choices[0].delta.content or ""
            
            output_tokens = count_tokens(response)
            total_cost += estimate_cost(tokens, output_tokens)

            if response.strip().lower() == category.lower():
                correct_predictions += 1
            
            time.sleep(delay)

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    return accuracy, total_cost
