import os
import json
from evaluate import load
from dataload.dataload import DatasetLoader as dl


# Define the MultiRC metric calculation function.
def multirc_metric(predictions, references):
    """
    Computes the MultiRC metric (Accuracy and F1) for the predictions.

    This function uses the `evaluate` library to calculate both accuracy and F1 scores
    based on the provided predictions and references.

    Args:
        predictions (list): A list of prediction dictionaries.
                            Each dictionary should have an 'idx' key (with nested indices) and a 'prediction' key.
        references (list): A list of reference labels as integers.

    Returns:
        dict: A dictionary containing the computed metrics (e.g., accuracy and F1 scores).
    """
    metric = load("super_glue", "multirc")
    results = metric.compute(predictions=predictions, references=references)
    return results


# Extended compute_multirc method.
def compute_multirc(model, subset_length=10):
    """
    Evaluates a model on a subset of the MultiRC dataset from SuperGLUE and saves the results to a file.

    This function loads a subset of the MultiRC dataset, uses the model to predict whether a given answer is correct,
    and then saves both the predictions and reference answers.

    Args:
        model (function): A function that calls the model and returns an answer for the given input text.
        subset_length (int): The number of entries from the MultiRC dataset to use for evaluation.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    # Load the MultiRC dataset.
    dataset = dl.load_dataset("multirc")  # Instantiate the MultiRC dataset loader.
    subset = dataset[:subset_length]

    # Initialize lists for storing predictions and reference labels.
    predictions = []
    references = []

    # Iterate over the subset of the dataset.
    for entry in subset:
        passage = entry['paragraph']
        question = entry['question']
        answer = entry['answer']
        label = entry['label']  # 0 = "incorrect", 1 = "correct"

        # Create the input text for the model.
        input_text = (
            f"Passage: {passage}\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Is the answer correct? Answer with either 0 for 'no' or 1 for 'yes'. Do not answer with text!"
        )

        try:
            # Call the model with the input text.
            response = model(input_text, max_new_tokens=50)
        except Exception as e:
            # If an error occurs during the model call, print the model's name, error, and entry details, then skip this entry.
            print(model.model_name)
            print(e)
            print(entry)
            continue

        try:
            # Convert the model's response to an integer value.
            prediction_label = int(response.strip())
        except Exception as e:
            # If the model's response is not in the correct format, print an error and skip this entry.
            print(model.model_name)
            print(response)
            print("Incorrect response format - skipping entry")
            print(e)
            continue

        # Save the prediction in the required format.
        predictions.append({
            'idx': {
                'answer': entry['idx']['answer'],
                'paragraph': entry['idx']['paragraph'],
                'question': entry['idx']['question']
            },
            'prediction': prediction_label
        })

        # Save the reference label.
        references.append(label)

    # Combine predictions and references into a single output dictionary.
    output_data = {
        "predictions": predictions,
        "references": references
    }

    # Create a dynamic directory for saving the results for the specific model.
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file path.
    output_file = os.path.join(output_dir, "multirc_predictions_references.json")

    # Write the output data to a JSON file.
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Predictions and references have been saved to {output_file}.")

    # Compute the MultiRC metrics using the predictions and references.
    results = multirc_metric(predictions, references)

    return results


def multirc_eval(models, subset_length=10):
    """
    Performs the MultiRC evaluation for all provided models and saves the results to a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of entries from the MultiRC dataset to use for each model.
    """
    all_results = {}
    for model in models:
        result = compute_multirc(model, subset_length)
        all_results[model.model_name] = result

    # Create a directory for final results.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)

    # Define the output file path for overall results.
    results_file = os.path.join(results_dir, "multirc_all_models_results.json")

    # Write all results to a JSON file.
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All MultiRC results have been saved to {results_file}.")

    return all_results
