import os
import json
import numpy as np

from Metrics.bleu import bleu_metric
from Metrics.llm_summarization import llm_sum_metric
from Metrics.meteor import meteor_metric
from Metrics.moverscore import moverscore_metric
from Metrics.rouge import rouge_metric
from dataload.dataload import DatasetLoader as dl


# Extended summarization evaluation function
def compute_sum(model, all_models, subset_length=10):
    """
    Evaluates a model on a subset of the CNN/DailyMail dataset and saves the predictions and references.

    This function loads a subset of the CNN/DailyMail dataset, uses the provided model to generate a summary
    for each article, and stores the predictions along with the reference summaries and article IDs.
    It then computes various summarization metrics (LLM-Sum, Rouge, Bleu, Meteor, MoverScore) and aggregates
    additional scores from an LLM-EVAL judge.

    Args:
        model (function): A function that calls the model and returns a summary for the given text.
        all_models (list): A list of models to be used as judges for evaluation.
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        dict: A dictionary containing computed metrics for the summarization task.
    """
    # Load the CNN/DailyMail dataset.
    dataset = dl.load_dataset("cnn_dailymail")
    subset = dataset[:subset_length]

    # Initialize lists to store model predictions, reference summaries, articles, and unique IDs.
    predictions = []
    references = []
    articles = []
    ids = []

    # Iterate over the dataset subset.
    for entry in subset:
        article = entry['article']
        reference_summary = entry['highlights']
        ids.append(entry['id'])

        # Create the input text for summarization.
        input_text = f"Summarize the following article:{article}. Your answer should not be longer than 50 words."

        try:
            # Get the model's summary.
            response = model(input_text, max_new_tokens=125)
        except Exception as e:
            print(model.model_name)
            print(e)
            print(entry)
            continue

        # Append the generated summary, article, and reference summary.
        predictions.append(response.strip())
        articles.append(article)
        references.append(reference_summary.strip())

    # Save predictions and references to a JSON file.
    output_data = {
        "id": ids,
        "predictions": predictions,
        "references": references
    }
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "summarization_predictions_references.json")
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)
    print(f"Predictions and references have been saved to {output_file}.")

    results = {}
    results["LLM_Sum"] = {}
    # Compute LLM-Sum metrics for summarization using each judge model.
    for judge in all_models:
        # llm_sum_metric returns a tuple: (evaluation results, errors)
        results["LLM_Sum"][judge.model_name], errors = llm_sum_metric(predictions, references, articles, ids, judge)
        print(f"Evaluation of {model.model_name} by {judge.model_name} completed")

        # Write any errors encountered during evaluation to a separate JSON file.
        error_dir = f"Errors/llm_sum_Errors/Errors_{model.model_name}/{judge.model_name}"
        os.makedirs(error_dir, exist_ok=True)
        error_file = os.path.join(error_dir, "errors.json")
        with open(error_file, "w") as file:
            json.dump(errors, file, indent=4)

    # Compute additional summarization metrics.
    results["Rouge"] = rouge_metric(predictions, references)
    results["Bleu"] = bleu_metric(predictions, references)
    results["Meteor"] = meteor_metric(predictions, references)
    results["Moverscore"] = moverscore_metric(predictions, references)

    # Process Moverscore: multiply by 100 for scaling.
    moverscore = results["Moverscore"]["moverscore"] * 100
    # Calculate average Rouge: average of rouge1, rouge2, rougeL, and rougeLsum, scaled to 0-100.
    rouge = results["Rouge"]
    avg_rouge = ((rouge["rouge1"] + rouge["rouge2"] + rouge["rougeL"] + rouge["rougeLsum"]) / 4) * 100

    # Calculate average LLM_Sum scores per judge.
    llm_sum_entries = results["LLM_Sum"]
    judge_scores = []

    for judge_name, judge_entry in llm_sum_entries.items():
        if isinstance(judge_entry, list):
            # Initialize lists to collect scores for each metric.
            coherence_values = []
            consistency_values = []
            grammar_values = []
            relevance_values = []
            fluency_values = []
            overall_score_values = []

            # Iterate over all evaluations from the current judge.
            for entry in judge_entry:
                if isinstance(entry, str):
                    try:
                        entry = json.loads(entry)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse entry from Judge {judge_name}. Entry: {entry}")
                        continue

                if "coherence" in entry:
                    coherence_values.append(entry["coherence"])
                else:
                    print(f"Warning: 'coherence' value missing from Judge {judge_name}")

                if "consistency" in entry:
                    consistency_values.append(entry["consistency"])
                else:
                    print(f"Warning: 'consistency' value missing from Judge {judge_name}")

                if "grammar" in entry:
                    grammar_values.append(entry["grammar"])
                else:
                    print(f"Warning: 'grammar' value missing from Judge {judge_name}")

                if "relevance" in entry:
                    relevance_values.append(entry["relevance"])
                else:
                    print(f"Warning: 'relevance' value missing from Judge {judge_name}")

                if "fluency" in entry:
                    fluency_values.append(entry["fluency"])
                else:
                    print(f"Warning: 'fluency' value missing from Judge {judge_name}")

                if "overall_score" in entry:
                    overall_score_values.append(entry["overall_score"])
                else:
                    print(f"Warning: 'overall_score' value missing from Judge {judge_name}")

            # Calculate average scores for the current judge.
            judge_score = {
                "judge_name": judge_name,
                "coherence": np.mean(coherence_values) if coherence_values else 0,
                "consistency": np.mean(consistency_values) if consistency_values else 0,
                "grammar": np.mean(grammar_values) if grammar_values else 0,
                "relevance": np.mean(relevance_values) if relevance_values else 0,
                "fluency": np.mean(fluency_values) if fluency_values else 0,
                "overall_score": np.mean(overall_score_values) if overall_score_values else 0
            }
            judge_scores.append(judge_score)

    if judge_scores:
        # Calculate average scores across all judges.
        avg_coherence = np.mean([judge["coherence"] for judge in judge_scores])
        avg_consistency = np.mean([judge["consistency"] for judge in judge_scores])
        avg_grammar = np.mean([judge["grammar"] for judge in judge_scores])
        avg_relevance = np.mean([judge["relevance"] for judge in judge_scores])
        avg_fluency = np.mean([judge["fluency"] for judge in judge_scores])
        avg_overall_score = np.mean([judge["overall_score"] for judge in judge_scores])

        avg_llm_sum = (avg_coherence + avg_consistency + avg_grammar + avg_relevance + avg_fluency) / 5

        # Calculate a weighted combined score.
        combined_score = (5 * avg_llm_sum + avg_rouge + moverscore) / 7

        results["judge_scores"] = judge_scores
        results["Final_scores"] = {
            "Average_Rouge": avg_rouge,
            "Moverscore": moverscore,
            "Average_LLM_Sum": avg_llm_sum,
            "Average_Coherence": avg_coherence,
            "Average_Consistency": avg_consistency,
            "Average_Grammar": avg_grammar,
            "Average_Relevance": avg_relevance,
            "Average_Fluency": avg_fluency,
            "Average_Overall_Score": avg_overall_score,
            "Combined_Score": combined_score,
        }

    return results


def summarization_eval(models, subset_length=10):
    """
    Evaluates summarization performance for all provided models and saves the aggregated results in a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of dataset entries to use for evaluation.
    """
    all_results = {}
    for model in models:
        result = compute_sum(model, models, subset_length)
        all_results[model.model_name] = result

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "summarization_benchmark_all_models_results.json")

    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All summarization results have been saved to {results_file}.")
