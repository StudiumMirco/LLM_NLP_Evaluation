import os
import json
from evaluate import load
from dataload.dataload import DatasetLoader as dl


# Define the BLEU metric calculation function.
def bleu_metric(predictions, references):
    """
    Computes the BLEU metric for the given predictions.

    This function loads the BLEU metric from the evaluate library and computes the BLEU scores based on the
    provided predicted translations and reference translations.

    Args:
        predictions (list): A list of predicted translations (strings).
        references (list): A list of reference translations, where each reference is provided as a list of strings.

    Returns:
        dict: A dictionary containing the BLEU metric results.
    """
    metric = load("bleu")
    results = metric.compute(predictions=predictions, references=references)
    return results


# Extended translation evaluation function for BLEU.
def compute_bleu(model, original_language, new_language, subset_length=10):
    """
    Evaluates a translation model on a subset of the WMT16 dataset and saves the predictions and references to a file.

    This function loads a subset of the WMT16 dataset (e.g., German-English), uses the provided model to generate
    translations, and then saves the predicted translations along with the reference translations in a JSON file.
    Finally, it computes the BLEU metric for the translations.

    Args:
        model (function): A function that calls the model and returns a translation for the given input text.
        original_language (str): The source language code (e.g., "de" for German).
        new_language (str): The target language code (e.g., "en" for English).
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        dict: A dictionary containing the BLEU metric results.
    """
    # Load the WMT16 dataset (e.g., for German-English translation).
    dataset = dl.load_dataset("wmt16")  # Instantiate the WMT16 dataset loader.
    subset = dataset[:subset_length]

    predictions = []  # List to store model-generated translations.
    references = []  # List to store reference translations.

    # Loop over the selected subset of the dataset.
    for entry in subset:
        print(entry)
        # Extract the source text and the reference translation.
        source_text = entry["translation"][original_language]  # e.g., German text.
        reference_translation = entry["translation"][new_language]  # e.g., English translation.

        # Create the input prompt for the model.
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
            # Call the model with the input prompt.
            response = model(input_text, max_new_tokens=150)
        except Exception as e:
            print(model.model_name)
            print(e)
            print(entry)
            continue

        # Append the trimmed response to the predictions list.
        predictions.append(response.strip())
        # Append the reference translation to the references list, wrapped in a list (as BLEU expects lists of references).
        references.append([reference_translation.strip()])

    # Save the predictions and references in a JSON file.
    output_data = {
        "predictions": predictions,
        "references": references
    }
    # Create a dynamic directory for saving results specific to the model.
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    # Choose output filename based on the original language.
    if original_language == "de":
        output_file = os.path.join(output_dir, "blue_predictions_references_de_en.json")
    else:
        output_file = os.path.join(output_dir, "blue_predictions_references_en_de.json")

    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Predictions and references have been saved to {output_file}.")

    # Compute the BLEU metric for the translation task.
    results = bleu_metric(predictions, references)
    return results


def bleu_eval(models, subset_length=10):
    """
    Evaluates translation performance for all provided models using the BLEU metric and saves the results to JSON files.

    This function evaluates models for German-to-English and English-to-German translation tasks separately,
    and saves the aggregated results in corresponding JSON files.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of dataset entries to use for evaluation.
    """
    # Evaluate for German-to-English translation.
    all_results = {}
    for model in models:
        result = compute_bleu(model, "de", "en", subset_length)
        all_results[model.model_name] = result

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "bleu_all_models_results_de_en.json")

    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    # Evaluate for English-to-German translation.
    all_results = {}
    for model in models:
        result = compute_bleu(model, "en", "de", subset_length)
        all_results[model.model_name] = result

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "bleu_all_models_results_en_de.json")

    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All translation results have been saved to {results_file}.")
