import os
import json
from evaluate import load
from dataload.dataload import DatasetLoader as dl


def bleurt_metric(predictions, references):
    """
    Computes the BLEURT metric for the given predictions.

    This function loads the BLEURT metric from the evaluate library (using the BLEURT module)
    and computes the metric scores based on the provided predictions and reference translations.
    It also calculates an average score from the list of individual scores.

    Args:
        predictions (list): A list of predicted translations as strings.
        references (list): A list of reference translations as strings.

    Returns:
        dict: A dictionary containing the BLEURT metric results, including an average score.
    """
    # Load the BLEURT metric
    metric = load("bleurt", module_type="metric")
    # Compute metric scores for each prediction-reference pair
    results = metric.compute(predictions=predictions, references=references)

    # Calculate the average BLEURT score from the list of scores
    average_score = sum(results['scores']) / len(results['scores'])
    results['average_score'] = average_score

    return results


def compute_bleurt(model, original_language, new_language, subset_length=10):
    """
    Evaluates a translation model on a subset of the WMT16 dataset using BLEURT.

    This function loads a subset of the WMT16 dataset, uses the provided model to generate translations,
    and saves both the model predictions and reference translations into a JSON file.
    Finally, it computes the BLEURT metric for the predictions.

    Args:
        model (function): A function that calls the model and returns a translation for the input text.
        original_language (str): The source language code (e.g., "de" for German).
        new_language (str): The target language code (e.g., "en" for English).
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        dict: A dictionary containing the BLEURT metric results.
    """
    # Load the WMT16 dataset (e.g., for German-English translation)
    dataset = dl.load_dataset("wmt16")
    subset = dataset[:subset_length]

    predictions = []  # List to store the model-generated translations
    references = []  # List to store the reference translations

    # Iterate over the dataset subset
    for entry in subset:
        # Debug print of the current entry (can be removed in production)
        print(entry)
        # Extract the source text and the reference translation based on the provided language codes
        source_text = entry["translation"][original_language]
        reference_translation = entry["translation"][new_language]

        # Create the input prompt for the model based on the original language
        if original_language == "de":
            input_text = (
                f"Translate the following German text into English: {source_text}. "
                "Answer just with the translation"
            )
        else:
            input_text = (
                f"Translate the following English text into German: {source_text}. "
                "Answer just with the translation"
            )

        try:
            # Call the model with the input prompt; allow up to 150 new tokens
            response = model(input_text, max_new_tokens=150)
        except Exception as e:
            # If an error occurs, print the model name, error, and the entry, then skip this entry
            print(model.model_name)
            print(e)
            print(entry)
            continue

        # Append the trimmed response and the reference translation to their respective lists
        predictions.append(response.strip())
        references.append(reference_translation.strip())

    # Prepare the output data dictionary
    output_data = {
        "predictions": predictions,
        "references": references
    }

    # Create a dynamic output directory based on the model's name
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Choose the output filename based on the original language
    if original_language == "de":
        output_file = os.path.join(output_dir, "bluert_predictions_references_de_en.json")
    else:
        output_file = os.path.join(output_dir, "bluert_predictions_references_en_de.json")

    # Write the output data to a JSON file
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)
    print(f"Predictions and references have been saved to {output_file}.")

    # Compute the BLEURT metric using the predictions and references
    results = bleurt_metric(predictions, references)

    return results


def bleurt_eval(models, subset_length=10):
    """
    Evaluates translation performance for all provided models using BLEURT and saves the results.

    This function iterates over a list of models, evaluates each model on a subset of the WMT16 dataset
    for German-to-English translation, and saves the aggregated BLEURT metric results into a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of dataset entries to use for each model's evaluation.
    """
    all_results = {}
    # Evaluate each model for German-to-English translation
    for model in models:
        result = compute_bleurt(model, "de", "en", subset_length)
        all_results[model.model_name] = result

    # Determine the directory for saving overall results
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)

    # Define the results filename
    results_file = os.path.join(results_dir, "bleurt_all_models_results_de_en.json")

    # Write the aggregated results to a JSON file
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)
    print(f"All translation results have been saved to {results_file}.")
