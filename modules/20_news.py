import os
import random
import time
from utils import client, count_tokens, estimate_cost

few_shot_examples = [
    ("From: bennett@kuhub.cc.ukans.edu\nSubject: Smoker's Lungs\nLines: 3\n\nHow long does it take a smoker's lungs to clear of the tar after quitting?", "sci.med"),
    ("From: jyoung@Cadence.COM (John Young)\nSubject: FFL&gunsmithing questions\nLines: 8\n\nHow would someone get a dealer's license and learn gunsmithing?", "talk.politics.guns")
]

def load_20news(dataset_path, num_samples=10):
    category_samples = {}
    categories = []

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            categories.append(category)
            files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
            selected_files = random.sample(files, min(num_samples, len(files)))
            category_samples[category] = [open(os.path.join(category_path, f), 'r', encoding='utf-8').read() for f in selected_files]

    return category_samples, categories

def prompt_20news(data, classes, model, prompt_type, pricing, token_budget=4000, delay=2):
    correct_predictions = 0
    total_cost = 0
    total_samples = 0
    class_list = ", ".join(classes)
    few_shot_prompt = "\n\n".join([f"Text: {ex[0]}\nCategory: {ex[1]}" for ex in few_shot_examples])

    for category, samples in data.items():
        for text in samples:
            if prompt_type == "direct_prompting":
                prompt = (f"{text}\n\n"
                           f"Classify this news article into one of the following newsgroups: "
                           f"{class_list}. Your answer should be exactly one of these newsgroups with no explanation.")
            elif prompt_type == "chain_of_thought":
                prompt = (f"{text}\n\n"
                           f"Classify this text into one of the following newsgroups:"
                           f"{class_list}.\n"
                           f"Think step by step to determine the correct newsgroup."
                           f"First, Analyze the content and identify the key topic or subject matter of the text"
                           f"Extract specific keywords, themes, or phrases that indicate the category of the text. "
                           f"Then, compare the topic and the detail to the provided newsgroups and select the most suitable one."
                           f"Finally, provide your answer prefixed with 'Final Answer: [newsgroups]'.")
            elif prompt_type == "few_shot":
                prompt = (f"Here are some examples of newsgroup posts and their categories:\n\n"
                           f"{few_shot_prompt}\n\n"
                           f"Now classify the following post into one of the following newsgroups without explanation: "
                           f"{', '.join(classes)}.\n"
                           f"Text: {text}\n"
                           f"(Your answer should be exactly one word/newsgroups)Newsgroup: ")
                
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
