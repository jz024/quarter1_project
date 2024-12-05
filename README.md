# LLM Text Classification Using Different Prompting Techniques

This project demonstrates how to classify text data from three different datasets—**20 News Groups**, **DBpedia**, and **Yelp Reviews**—using large language models (LLMs) with various prompting techniques, including **direct prompting**, **chain of thought (CoT)**, and **few-shot prompting**. The project dynamically handles datasets and prompting techniques for maximum flexibility.

## Project Overview

### Supported Datasets
1. **20 News Groups**:
   - A collection of approximately 20,000 newsgroup documents across 20 categories.
2. **DBpedia**:
   - A large-scale ontology dataset derived from Wikipedia, containing classified text data.
3. **Yelp Reviews**:
   - Customer reviews with star ratings (1-5) from Yelp.

### Prompting Techniques
- **Direct Prompting**: Asks the LLM to classify the text without additional reasoning or examples.
- **Chain of Thought (CoT)**: Guides the LLM to reason step-by-step before providing an answer.
- **Few-Shot Prompting**: Provides examples of text and their corresponding classifications before asking the LLM to classify new text.

---

## Setup and Installation

### Step 1: Clone the Repository
Clone this repository to your local machine:
```bash
git clone <repository_url>
cd <repository_name>
```

### Step 2: Install Dependencies

Install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Use the ```run.py``` script to classify datasets using different prompting techniques and models. The script dynamically selects the dataset and prompting function based on the user’s input.

### Commend Syntax
```bash
python run.py --dataset <dataset_name> --prompting <prompting_technique> --model <model_name>
```

### Example

#### Run Chain-of-Thought on Yelp Dataset with 8B Model:
```bash
python run.py --dataset yelp --prompting chain_of_thought --model Meta-Llama-3.1-8B-Instruct
```



