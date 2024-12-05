import os
import random
import time
from utils import client, count_tokens, estimate_cost
import csv
import re

few_shot_examples = [
    (
        "Top notch doctor in a top notch practice. Can't say I am surprised when I was referred to him by another doctor who I think is wonderful and because he went to one of the best medical schools in the country. It is really easy to get an appointment. There is minimal wait to be seen and his bedside manner is great.",
        "5"
    ),
    (
        "This place is absolute garbage... Half of the tees are not available, including all the grass tees. It is cash only, and they sell the last bucket at 8, despite having lights. And if you finish even a minute after 8, don't plan on getting a drink. The vending machines are sold out (of course) and they sell drinks inside, but close the drawers at 8 on the dot. There are weeds grown all over the place. I noticed some sort of batting cage, but it looks like those are out of order as well. Someone should buy this place and turn it into what it should be.",
        "1"
    )
]

def load_yelp(dataset_path, num_samples=10):
    category_samples = {str(i): [] for i in range(1, 6)}

    with open(dataset_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            class_index, review_text = row
            if class_index in category_samples:
                category_samples[class_index].append(review_text)

    selected_samples = {}
    for class_index, reviews in category_samples.items():
        selected_samples[class_index] = random.sample(reviews, min(num_samples, len(reviews)))
    category = ['1','2','3','4','5']
        
    return selected_samples, category

def prompt_yelp(data, classes, model, prompt_type, pricing, token_budget=4000, delay=2):
    correct_predictions = 0
    total_cost = 0
    total_samples = 0
    class_list = ", ".join(classes)
    few_shot_prompt = "\n\n".join([f"Review: {ex[0]}\nRating: {ex[1]}" for ex in few_shot_examples])

    for rating, samples in data.items():
        for text in samples:
            if prompt_type == "direct_prompting":
                prompt = (f"{text}\n\n"
                          f"Classify this review into one of the following star ratings: {class_list}. "
                          f"Your answer should be exactly one of these ratings with no explanation.")
            elif prompt_type == "chain_of_thought":
                prompt = (f"{text}\n\n"
                          f"Classify this text into one of the following ratings: {class_list}.\n"
                          "Think step by step to determine the correct rating. "
                          "First, identify the overall sentiment of the review: Is it positive, negative, or neutral? "
                          "Then, highlight any specific details or phrases that indicate the level of satisfaction (e.g., mentions of service, food quality, ambiance, or specific issues). "
                          "Consider whether the review describes a highly positive experience (5 stars), a good but imperfect experience (4 stars), an average experience (3 stars), "
                          "a disappointing experience (2 stars), or a very negative experience (1 star). "
                          "Finally, make sure that your answer is one of the rating categories provided, and provide your answer prefixed with 'Final Answer: [rating]'.")
            elif prompt_type == "few_shot":
                prompt = (f"Here are some examples of reviews and their ratings:\n\n{few_shot_prompt}\n\n"
                          f"Now classify the following review to one of the following ratings without explanation: {class_list}.\n"
                          f"Review: {text}\nRating: ")

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
                        final_answer = re.sub(r"[^a-z0-9]", "", final_answer)
                        break

                if final_answer == rating.lower():
                    correct_predictions += 1
            else:
                if response.strip().lower() == rating.lower():
                    correct_predictions += 1

            output_tokens = count_tokens(response)
            total_cost += estimate_cost(tokens, output_tokens, pricing["input_price_per_million"], pricing["output_price_per_million"])
            
            time.sleep(delay)

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    return accuracy, total_cost