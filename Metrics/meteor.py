import os
import json
from evaluate import load
from dataload.dataload import DatasetLoader as dl


# Define the METEOR metric calculation function.
def meteor_metric(predictions, references):
    """
    Computes the METEOR metric for the given predictions.

    This function uses the `evaluate` library to compute the METEOR score based on the provided
    predictions and references.

    Args:
        predictions (list): A list of predicted translations as strings.
        references (list): A list of reference translations, where each reference is a list of strings.

    Returns:
        dict: A dictionary containing the computed METEOR score.
    """
    metric = load("meteor")
    results = metric.compute(predictions=predictions, references=references)
    return results


# Extended translation evaluation method using METEOR.
def compute_meteor(model, original_language, new_language, subset_length=10):
    """
    Evaluates a model on a subset of the WMT16 dataset and saves the results to a file.

    This function loads a subset of the WMT16 dataset, uses the model to generate translations,
    and saves both the model predictions and the reference translations. It then computes the
    METEOR metric for the translations.

    Args:
        model (function): A function that calls the model and returns a translation for the given text.
        original_language (str): The language code of the source text (e.g., "de" for German).
        new_language (str): The language code for the target translation (e.g., "en" for English).
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        dict: A dictionary containing the computed METEOR metric.
    """
    # Load the WMT16 dataset (e.g., German-English).
    dataset = dl.load_dataset("wmt16")
    subset = dataset[:subset_length]

    # Initialize lists to store predictions and references.
    predictions = []
    references = []

    # Loop over the selected subset of the dataset.
    for entry in subset:
        # Retrieve the source text and its reference translation.
        source_text = entry["translation"][original_language]  # Source text (e.g., German)
        reference_translation = entry["translation"][new_language]  # Reference translation (e.g., English)

        # Create the input text for the model based on the source language.
        if original_language == "de":
            input_text = (
                f"Translate the following German text into English: {source_text}. "
                "Answer just with the translation."
            )
        else:
            input_text = (
                f"Translate the following English text into German: {source_text}. "
                "Answer just with the translation."
            )

        try:
            # Call the model with the input text, requesting up to 75 new tokens.
            response = model(input_text, max_new_tokens=75)
        except Exception as e:
            # If an error occurs, print the model name, error details, and the current entry, then skip this entry.
            print(model.model_name)
            print(e)
            print(entry)
            continue

        # Append the trimmed model response to the predictions list.
        predictions.append(response.strip())

        # Append the trimmed reference translation to the references list.
        # METEOR expects each reference to be provided as a list.
        references.append([reference_translation.strip()])

    # Prepare the output data containing the predictions and references.
    output_data = {
        "predictions": predictions,
        "references": references
    }

    # Create a directory specific to the model for saving results.
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Set the output filename based on the source language.
    if original_language == "de":
        output_file = os.path.join(output_dir, "meteor_predictions_references_de_en.json")
    else:
        output_file = os.path.join(output_dir, "meteor_predictions_references_en_de.json")

    # Write the predictions and references to a JSON file.
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Predictions and references have been saved to {output_file}.")

    # Compute the METEOR metric using the predictions and references.
    results = meteor_metric(predictions, references)

    return results


def meteor_eval(models, subset_length=10):
    """
    Performs translation evaluation using the METEOR metric for all provided models and saves the results to JSON files.

    This function evaluates the models for both German-to-English and English-to-German translation tasks.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of dataset entries to use for each evaluation.
    """
    # Evaluate for German-to-English translation.
    all_results = {}
    for model in models:
        result = compute_meteor(model, "de", "en", subset_length)
        all_results[model.model_name] = result

    # Create a directory for final results.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "meteor_all_models_results_de_en.json")

    # Save the German-to-English results.
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    # Evaluate for English-to-German translation.
    all_results = {}
    for model in models:
        result = compute_meteor(model, "en", "de", subset_length)
        all_results[model.model_name] = result

    # Create a directory for final results (if not already created).
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "meteor_all_models_results_en_de.json")

    # Save the English-to-German results.
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All translation results have been saved to {results_file}.")
