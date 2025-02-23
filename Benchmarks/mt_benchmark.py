import os
import json

from Metrics.bleu import bleu_metric
from Metrics.bleurt import bleurt_metric
from Metrics.meteor import meteor_metric
from dataload.dataload import DatasetLoader as dl


# Extended compute_translation method
def compute_mt(model, original_language, new_language, subset_length=10):
    """
    Evaluates a model on a subset of the WMT16 dataset and saves the results to a file.

    This function loads a subset of the WMT16 dataset, uses the model to generate translations,
    and saves both the model predictions and the reference translations.

    Args:
        model (function): A function that calls the model and returns a translation for the given text.
        original_language (str): The source language code (e.g., "de" for German).
        new_language (str): The target language code (e.g., "en" for English).
        subset_length (int): The number of entries from the WMT16 dataset to use for evaluation.

    Returns:
        dict: A dictionary containing the computed metrics.
    """

    # Load the WMT16 dataset (German-English)
    dataset = dl.load_dataset("wmt16")  # Instantiation of the WMT16 dataset (German-English)
    subset = dataset[:subset_length]

    # Lists to store the results
    predictions = []
    references = []
    # Alternative formatting for BleuRT
    referencesrt = []

    # Loop over the subset of the dataset
    for entry in subset:

        source_text = entry["translation"][original_language]  # Source text (e.g., German)
        reference_translation = entry["translation"][new_language]  # Reference translation (e.g., English)

        # Create the input text
        if original_language == "de":
            input_text = f"Translate the following German text into English: {source_text}. Answer just with the translation."
        else:
            input_text = f"Translate the following English text into German: {source_text}. Answer just with the translation."

        try:
            # Call the model with the input text
            response = model(input_text, max_new_tokens=150)
        # Catch model errors: if an error occurs, skip the current entry
        except Exception as e:
            print(model.model_name)
            print(e)
            print(entry)
            continue

        # Save the model's prediction as a stripped string
        predictions.append(response.strip())

        # Save the reference translation as a list (BLEU expects a list of references)
        references.append([reference_translation.strip()])
        referencesrt.append(reference_translation.strip())

    # Save the predictions and references in a JSON file - the file naming is based on the model ID
    output_data = {
        "predictions": predictions,
        "references": references
    }

    # Dynamic path for saving the results in a model-specific folder
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    # Set the output filename based on the source language
    if original_language == "de":
        output_file = os.path.join(output_dir, "mt_predictions_references_de_en.json")
    else:
        output_file = os.path.join(output_dir, "mt_predictions_references_en_de.json")

    # Write the output data to a JSON file
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Predictions and references have been saved to {output_file}.")

    results = {}

    if original_language == "de":

        # Compute translation metrics
        bleu = bleu_metric(predictions, references)
        bleu["bleu_norm"] = bleu["bleu"] * 100
        meteor = meteor_metric(predictions, references)
        meteor["meteor_norm"] = meteor["meteor"] * 100
        bleurt = bleurt_metric(predictions, referencesrt)
        bleurt["average_score_norm"] = bleurt["average_score"] * 100

        # Store translation metrics
        results["Bleu"] = bleu
        results["Meteor"] = meteor
        results["BleuRT"] = bleurt
        results["Gesamtscore"] = (bleu["bleu_norm"] + meteor["meteor_norm"] + bleurt["average_score_norm"]) / 3

    # BLEURT only supports the target language English
    else:
        # Compute translation metrics
        bleu = bleu_metric(predictions, references)
        bleu["bleu_norm"] = bleu["bleu"] * 100
        meteor = meteor_metric(predictions, references)
        meteor["meteor_norm"] = meteor["meteor"] * 100

        results["Bleu"] = bleu
        results["Meteor"] = meteor
        results["Gesamtscore"] = (bleu["bleu_norm"] + meteor["meteor_norm"]) / 2

    return results


def mt_eval(models, subset_length=10):
    """
    Performs translation evaluation for all models and saves the results to a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of dataset entries to use for evaluation.
    """

    # German-to-English evaluation
    all_results = {}
    for model in models:
        result_de_en = compute_mt(model, "de", "en", subset_length)
        result_en_de = compute_mt(model, "en", "de", subset_length)

        # Compute the final overall score as the average of both directions
        final_score = (result_de_en["Gesamtscore"] + result_en_de["Gesamtscore"]) / 2
        result_de_en["FinalGesamtscore"] = final_score
        result_en_de["FinalGesamtscore"] = final_score

        all_results[model.model_name] = {
            "de_en": result_de_en,
            "en_de": result_en_de
        }

    # Create a folder for final results and save the aggregated results to a JSON file
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "mt_all_models_results_de_en.json")

    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All translation results have been saved to {results_file}.")
