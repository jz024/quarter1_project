import argparse
import json
import importlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="Specify the dataset to use (e.g., '20_news', 'dbpedia', 'yelp')."
    )
    parser.add_argument(
        "--prompting", 
        type=str, 
        required=True, 
        help="Specify the prompting technique (e.g., 'direct_prompting', 'chain_of_thought', 'few_shot')."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Specify the LLM model to use (e.g., 'Meta-Llama-3.1-8B-Instruct')."
    )
    args = parser.parse_args()

    with open("config.json", "r") as f:
        config = json.load(f)

    dataset_name = args.dataset
    prompting_technique = args.prompting
    model = args.model

    if dataset_name not in config["datasets"]:
        raise ValueError(f"Invalid dataset '{dataset_name}'. Available options: {list(config['datasets'].keys())}.")
    if prompting_technique not in config["prompting_methods"]:
        raise ValueError(f"Invalid prompting technique '{prompting_technique}'. Available options: {config['prompting_methods']}.")
    if model not in config["models"]:
        raise ValueError(f"Invalid model '{model}'. Available options: {config['models']}.")

    dataset_module = importlib.import_module(f"modules.{dataset_name}")
    dataset_config = config["datasets"][dataset_name]
    if dataset_name == "dbpedia":
        data, categories = dataset_module.load_dbpedia(
            dataset_config["path"], 
            dataset_config["categories_path"], 
            dataset_config["num_samples"]
        )
    elif dataset_name == "yelp":
        data, categories = dataset_module.load_yelp(
            dataset_config["path"], 
            dataset_config["num_samples"]
        )
    else:
        data, categories = dataset_module.load_20news(
            dataset_config["path"], 
            dataset_config["num_samples"]
        )

    print(f"Running {prompting_technique} on {dataset_name} with {model}...")
    prompt_function = getattr(dataset_module, f"prompt_{dataset_name}")

    accuracy, total_cost = prompt_function(
        data,
        categories,
        model,
        prompting_technique,
        config["pricing"]
    )
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total Cost: ${total_cost:.2f}")

if __name__ == "__main__":
    main()

