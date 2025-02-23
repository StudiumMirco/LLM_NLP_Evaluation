import os
import json
from dataload.dataload import DatasetLoader as dl

from collections import defaultdict
from typing import List
import numpy as np
from moverscore_v2 import word_mover_score


def sentence_score(hypothesis: str, references: List[str], trace=0):
    """
    Computes the MoverScore for a single hypothesis against a list of reference summaries.

    This function uses the word mover score implementation to calculate the similarity
    between the hypothesis and each reference, then returns the average score.

    Args:
        hypothesis (str): The generated summary.
        references (List[str]): A list of reference summaries.
        trace (int, optional): If greater than 0, prints debugging information. Defaults to 0.

    Returns:
        float: The average MoverScore for the hypothesis against the references.
    """
    # Create default IDF dictionaries with a default value of 1.0.
    idf_dict_hyp = defaultdict(lambda: 1.0)
    idf_dict_ref = defaultdict(lambda: 1.0)

    # Repeat the hypothesis for each reference so that both lists have equal length.
    hypothesis = [hypothesis] * len(references)

    # Compute the word mover score for each hypothesis-reference pair.
    scores = word_mover_score(
        references,
        hypothesis,
        idf_dict_ref,
        idf_dict_hyp,
        stop_words=[],  # No stop words are removed.
        n_gram=1,  # Using unigrams for the comparison.
        remove_subwords=False,
    )

    # Calculate the average score.
    sentence_score = np.mean(scores)

    if trace > 0:
        print(hypothesis, references, sentence_score)

    return sentence_score


def compute_score(hyp, refs):
    """
    Computes the average MoverScore for a corpus.

    This function iterates over each hypothesis and its corresponding references,
    computes the sentence-level MoverScore, and then averages the scores across all hypotheses.

    Args:
        hyp (List[str]): List of generated summaries.
        refs (List[List[str]]): List of reference summaries for each generated summary.

    Returns:
        float: The average MoverScore for the corpus.
    """
    corpus_score = 0
    for i, j in enumerate(hyp):
        corpus_score += sentence_score(j, refs[i])
    return corpus_score / len(hyp)


# Define the MoverScore metric calculation function.
def moverscore_metric(predictions, references):
    """
    Computes the MoverScore metric for the given predictions and references.

    This function calculates the MoverScore by comparing each predicted summary with its
    corresponding reference summary, and returns the rounded score.

    Args:
        predictions (list): A list of generated summaries as strings.
        references (list): A list of reference summaries as strings.

    Returns:
        dict: A dictionary containing the MoverScore metric.
    """
    # Compute the overall corpus score using the helper function.
    moverscore = compute_score(hyp=predictions, refs=references)
    return {"moverscore": round(moverscore, 5)}


# Extended compute_summarization method for MoverScore evaluation.
def compute_moverscore(model, subset_length=10):
    """
    Evaluates a model on a subset of the CNN/DailyMail dataset using the MoverScore metric.

    This function loads a subset of the CNN/DailyMail dataset, uses the model to generate summaries,
    saves both the predictions and the reference summaries to a JSON file, and computes the MoverScore.

    Args:
        model (function): A function that calls the model and returns a summary for the given input text.
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        dict: A dictionary containing the computed MoverScore metric.
    """
    # Load the CNN/DailyMail dataset.
    dataset = dl.load_dataset("cnn_dailymail")  # Instantiate the dataset loader.
    subset = dataset[:subset_length]

    # Initialize lists for storing predictions and references.
    predictions = []
    references = []

    # Iterate over each entry in the dataset subset.
    for entry in subset:
        article = entry['article']  # The article text.
        reference_summary = entry['highlights']  # The reference summary.

        # Create the input text for the model.
        input_text = f"Summarize the following article:{article}"

        try:
            # Call the model with the input text, allowing up to 75 new tokens.
            response = model(input_text, max_new_tokens=75)
        except Exception as e:
            # If an error occurs during model inference, print the model name, error details, and entry, then skip this entry.
            print(model.model_name)
            print(e)
            print(entry)
            continue

        # Optionally print the response for debugging.
        print(response)

        # Append the trimmed model prediction and reference summary.
        predictions.append(response.strip())
        references.append(reference_summary.strip())

    # Prepare the output data containing predictions and references.
    output_data = {
        "predictions": predictions,
        "references": references
    }

    # Create a directory specific to the model to save the results.
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Define the output filename for saving the predictions and references.
    output_file = os.path.join(output_dir, "moverscore_predictions_references.json")

    # Write the output data to a JSON file.
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Predictions and references have been saved to {output_file}.")

    # Compute the MoverScore metric using the predictions and references.
    results = moverscore_metric(predictions, references)

    return results


def moverscore_eval(models, subset_length=10):
    """
    Evaluates multiple models on a subset of the CNN/DailyMail dataset using the MoverScore metric
    and saves the aggregated results to a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of dataset entries to use for each model.

    Returns:
        dict: A dictionary containing the MoverScore metrics for all evaluated models.
    """
    all_results = {}
    for model in models:
        result = compute_moverscore(model, subset_length)
        all_results[model.model_name] = result

    # Create a directory for final aggregated results.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)

    # Define the output file for the aggregated results.
    results_file = os.path.join(results_dir, "moverscore_all_models_results.json")

    # Write the aggregated results to a JSON file.
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All summarization results have been saved to {results_file}.")
