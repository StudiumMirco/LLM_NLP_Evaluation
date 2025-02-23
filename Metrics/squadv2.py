from evaluate import load
from dataload.dataload import DatasetLoader as dl
import json
import os


def squad_metric(predictions, references):
    """
    Calculates the SQuAD metric for a given set of predictions and references.

    This function loads the SQuAD metric and computes accuracy and F1 score based on the provided
    predictions and references.

    Args:
        predictions (list): A list of dictionaries containing model predictions.
                            Each dictionary includes the keys 'id' and 'prediction_text'.
        references (list): A list of dictionaries containing reference answers.
                           Each dictionary includes the keys 'id' and 'answers', where 'answers' is a
                           dictionary with 'answer_start' and 'text'.

    Returns:
        dict: A dictionary containing the computed metrics, including accuracy and F1 score.
    """
    # Initialize the SQuAD metric from the evaluate library.
    metric = load("squad")
    # Compute the metric using the given predictions and references.
    results = metric.compute(predictions=predictions, references=references)
    return results


def compute_squadv2(model, subset_length=10):
    """
    Evaluates a model on a subset of the SQuAD v2 dataset.

    This function loads a subset of the SQuAD v2 dataset and uses the provided model to answer questions.
    The results (predictions and reference answers) are saved as a JSON file and the SQuAD metric is computed.

    Args:
        model (function): A function that calls the model and returns an answer for the given text (e.g. gpt4o).
        subset_length (int): The number of entries from the SQuAD v2 dataset to use for evaluation (default is 10).

    Returns:
        dict: A dictionary containing the computed metrics, including accuracy and F1 score.
    """
    # Load the SQuAD v2 dataset using the dataset loader.
    dataset = dl.load_dataset("squad_v2")
    # Select a subset of the dataset based on the provided length.
    subset = dataset[0:subset_length]

    # Initialize lists to store the predictions and references.
    predictions = []
    references = []

    # Loop over each entry in the subset.
    for entry in subset:
        # Extract the question, context passage, and unique identifier from the dataset entry.
        question = entry['question']  # The question from the dataset.
        context = entry['context']  # The context passage from the dataset.
        id = entry['id']  # Unique identifier for the entry.

        # Create a formatted input string for the model API call.
        input_text = (
            f"Here is a context passage: {context}\n"
            f"Based on this context, answer the following question precisely:\n"
            f"Question: {question}\n"
            f"Answer only with the exact information from the context. If there is no suitable answer, respond with: \"No answer possible\"."
            f" Do not explain your answer!"
        )

        try:
            # Call the model with the input text and clean the response by keeping only alphanumeric characters and spaces.
            response = ''.join(
                char for char in model(input_text, max_new_tokens=50) if char.isalnum() or char.isspace()).strip()
        except Exception as e:
            # If an error occurs during the model call, print the model name, error, and entry details, then skip this entry.
            print(model.model_name)
            print(e)
            print(entry)
            continue

        # If the response indicates "No answer possible", set the response to an empty string.
        if response in ["No answer possible", "No answer possible.", "No answer possible\n"]:
            response = ""

        # Append the model's prediction to the predictions list in the required format.
        predictions.append({
            "id": id,
            "prediction_text": response
        })

        # Extract the reference answers and their starting positions from the dataset entry.
        if entry['answers']['text']:
            answer_texts = entry['answers']['text']  # List of answer texts.
            answer_starts = entry['answers']['answer_start']  # List of corresponding start positions.
        else:
            # If no answers are provided, set default values.
            answer_texts = [""]
            answer_starts = [-1]

        # Append the reference data in the required format.
        references.append({
            "id": id,
            "answers": {
                "answer_start": answer_starts,
                "text": answer_texts
            }
        })

    # Combine the predictions and references into a single dictionary.
    output_data = {
        "predictions": predictions,
        "references": references
    }

    # Create a dynamic directory path to save the results specific to the model.
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Define the output filename for saving the predictions and references.
    output_file = os.path.join(output_dir, "squad_predictions_references.json")

    # Write the output data to a JSON file.
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Predictions and references have been saved to {output_file}.")

    # Calculate the SQuAD metrics (accuracy and F1 score) using the predictions and references.
    results = squad_metric(predictions, references)

    return results


def squad_eval(models, subset_length=10):
    """
    Performs the SQuAD v2 evaluation for all provided models and saves the overall results to a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of entries from the SQuAD v2 dataset to use for each model.
    """
    # Initialize a dictionary to hold the results for all models.
    all_results = {}
    for model in models:
        # Evaluate the current model on the SQuAD v2 subset and store the computed metrics.
        result = compute_squadv2(model, subset_length)
        all_results[model.model_name] = result

    # Determine the current directory and set up the path for saving final results.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    # Define the output filename for the overall results.
    results_file = os.path.join(results_dir, "squad_all_models_results.json")

    # Write the overall results to a JSON file.
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All SQuAD results have been saved to {results_file}.")
