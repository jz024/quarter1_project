import os
import random
import time
from utils import client, count_tokens, estimate_cost
import csv

few_shot_examples = [
    (
        "Q-workshop Q-workshop is a Polish company located in Pozna≈Ñ that specializes in design "
        "and production of polyhedral dice and dice accessories for use in various games "
        "(role-playing games, board games, and tabletop wargames). They also run an online retail store and maintain "
        "an active forum community. Q-workshop was established in 2001 by Patryk Strzelewicz, a student from Pozna≈Ñ. "
        "Initially, the company sold its products via online auction services but in 2005 a website and online store were "
        "established.",
        "Company"
    ),
    (
        "USS Truxtun The first USS Truxtun was a brig in the United States Navy. She was named for Commodore Thomas "
        "Truxtun and was an active participant in the Mexican-American War.",
        "MeanOfTransportation"
    )
]

def load_dbpedia(dataset_path, classes_file, num_samples=10):
    with open(classes_file, 'r') as file:
        classes = [line.strip() for line in file.readlines()]
    
    index_to_class = {str(i + 1): cls for i, cls in enumerate(classes)}
    all_samples_by_class = {cls: [] for cls in classes}
    
    with open(dataset_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            class_index, title, content = row
            text = f"{title} {content}"
            category = index_to_class[class_index]
            all_samples_by_class[category].append(text)
    
    category_samples = {}
    for category, samples in all_samples_by_class.items():
        if len(samples) > num_samples:
            selected_samples = random.sample(samples, num_samples)
        else:
            selected_samples = samples
        category_samples[category] = selected_samples
    
    return category_samples, classes

def prompt_dbpedia(data, classes, model, prompt_type, pricing, token_budget=4000, delay=2):
    correct_predictions = 0
    total_cost = 0
    total_samples = 0
    class_list = ", ".join(classes)
    few_shot_prompt = "\n\n".join([f"Text: {ex[0]}\nCategory: {ex[1]}" for ex in few_shot_examples])

    for category, samples in data.items():
        for text in samples:
            if prompt_type == "direct_prompting":
                prompt = (f"{text}\n\n"
                          f"Classify this text into one of the following categories: {class_list}. "
                          f"Your answer should be exactly one of these categories with no explanation.")
            elif prompt_type == "chain_of_thought":
                prompt = (f"{text}\n\n"
                          f"Classify this text into one of the following categories: {class_list}.\n"
                          "Think step by step to determine the correct category. "
                          "First, analyze the content and identify the key topic or subject matter of the text. "
                          "Extract specific details or keywords that point to the text's main category. "
                          "Then, compare the topic and details to the provided categories and select the most suitable one. "
                          "Finally, provide your answer prefixed with 'Final Answer: [category]'.")
            elif prompt_type == "few_shot":
                prompt = (f"Here are some examples of texts and their categories:\n\n{few_shot_prompt}\n\n"
                          f"Now classify the following text into one of the following categories without explanation: "
                          f"{class_list}.\nText: {text}\nCategory: ")

            tokens = count_tokens(prompt)
            if tokens > token_budget:
                continue

            total_samples += 1

            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0.1,
                top_p=0.1
            )

            response = ""
            for chunk in completion:
                response += chunk.choices[0].delta.content or ""

            if prompt_type == "chain_of_thought":
                final_answer = None
                for line in response.split("\n"):
                    if "Final Answer:" in line:
                        final_answer = line.split("Final Answer:")[-1].strip().lower()
                        break
                if final_answer == category.lower():
                    correct_predictions += 1
            else:
                if response.strip().lower() == category.lower():
                    correct_predictions += 1

            output_tokens = count_tokens(response)
            total_cost += estimate_cost(tokens, output_tokens, pricing["input_price_per_million"], pricing["output_price_per_million"])
            
            time.sleep(delay)

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    return accuracy, total_cost