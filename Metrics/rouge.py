import os
import json
from evaluate import load
from dataload.dataload import DatasetLoader as dl


# Define the Rouge metric calculation function.
def rouge_metric(predictions, references):
    """
    Computes the ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum metrics for the predictions.

    This function loads the ROUGE metric from the evaluate library and calculates the scores
    based on the provided predictions and references.

    Args:
        predictions (list): A list of predicted summaries as strings.
        references (list): A list of reference summaries as strings.

    Returns:
        dict: A dictionary containing the computed ROUGE metrics.
    """
    metric = load("rouge")
    results = metric.compute(predictions=predictions, references=references)
    return results


# Extended compute_summarization method for ROUGE evaluation.
def compute_rouge(model, subset_length=10):
    """
    Evaluates a model on a subset of the CNN/DailyMail dataset and saves the results to a file.

    This function loads a subset of the CNN/DailyMail dataset, uses the model to generate a summary
    for each article, and saves both the model's predictions and the reference summaries.

    Args:
        model (function): A function that calls the model and returns a summary for the given text.
        subset_length (int): The number of entries from the CNN/DailyMail dataset to use for evaluation.

    Returns:
        dict: A dictionary containing the computed ROUGE metrics.
    """
    # Load the CNN/DailyMail dataset.
    dataset = dl.load_dataset("cnn_dailymail")  # Instantiate the CNN/DailyMail dataset loader.
    subset = dataset[:subset_length]

    # Initialize lists to store the predictions, references, and entry IDs.
    predictions = []
    references = []
    ids = []

    # Iterate over the subset of the dataset.
    for entry in subset:
        article = entry['article']  # The article text.
        reference_summary = entry['highlights']  # The reference summary.
        ids.append(entry['id'])  # Store the unique ID of the entry.

        # Create the input text for the model.
        input_text = f"Summarize the following article:{article}. Your answer should not be longer than 50 words."

        try:
            # Call the model with the input text and request up to 75 new tokens.
            response = model(input_text, max_new_tokens=75)
        except Exception as e:
            # If an error occurs during model prediction, print the model name, error, and entry details, then skip this entry.
            print(model.model_name)
            print(e)
            print(entry)
            continue

        # Append the trimmed model response to the predictions list.
        predictions.append(response.strip())
        # Append the trimmed reference summary to the references list.
        references.append(reference_summary.strip())

    # Prepare the output data including IDs, predictions, and references.
    output_data = {
        "id": ids,
        "predictions": predictions,
        "references": references
    }

    # Create a dynamic directory path to save the results in a model-specific folder.
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Define the output filename for saving the predictions and references.
    output_file = os.path.join(output_dir, "rouge_predictions_references.json")

    # Write the output data to a JSON file.
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Predictions and references have been saved to {output_file}.")

    # Compute the ROUGE metrics for the summarization task.
    results = rouge_metric(predictions, references)

    return results


def rouge_eval(models, subset_length=10):
    """
    Performs summarization evaluation with ROUGE on the CNN/DailyMail-Dataset for all provided models and saves the results to a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of entries from the CNN/DailyMail dataset to use for each model.
    """
    all_results = {}
    for model in models:
        result = compute_rouge(model, subset_length)
        all_results[model.model_name] = result

    # Create a directory for the final results.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    # Define the output filename for the overall ROUGE results.
    results_file = os.path.join(results_dir, "rouge_all_models_results.json")

    # Write the overall results to a JSON file.
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All summarization results have been saved to {results_file}.")
