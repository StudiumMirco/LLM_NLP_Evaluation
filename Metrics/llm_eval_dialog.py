import os
import json
import numpy as np

from dataload.dataload import DatasetLoader as dl


def llm_dialog_metric(predictions, dialogues, contexts, ids, judge):
    """
    Computes LLM-as-a-Judge metrics for dialogue responses using an LLM judge (e.g., GPT-4).

    This function constructs a system prompt that instructs the judge to evaluate the quality
    of the assistant's dialogue response based on various criteria (appropriateness, coherence, content,
    grammar, relevance, and overall score). It then calls the judge model on each dialogue instance and
    attempts to parse the output as JSON.

    Args:
        predictions (list): A list of predicted dialogue responses from the model.
        dialogues (list): A list of dialogue references or previous dialogue turns.
        contexts (list): A list of contextual information associated with each prediction.
        ids (list): A list of unique identifiers for each dialogue instance.
        judge (function): The LLM model used as a judge to evaluate the responses.

    Returns:
        tuple: A tuple containing:
            - evaluation_results (list): A list of evaluation results (each as a dictionary).
            - evaluation_errors (list): A list of error logs for entries where evaluation failed.
    """
    # Define the system prompt for the judge.
    system_prompt = (
        "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant in the context dialogue. "
        "You will be given context information, the previous dialogue and the assistant's answer. "
        "Your evaluation should focus on the assistant's answer. Be as objective as possible. "
        "The output should be formatted as a JSON instance that strictly conforms to the following JSON schema:\n"
        "{\"properties\": {\"appropriateness\": {\"title\": \"Appropriateness\", \"description\": \"appropriateness score in the range of 1 to 100\", \"type\": \"integer\"}, "
        "\"coherence\": {\"title\": \"Coherence\", \"description\": \"coherence score in the range of 1 to 100\", \"type\": \"integer\"}, "
        "\"content\": {\"title\": \"Content\", \"description\": \"content score in the range of 1 to 100\", \"type\": \"integer\"}, "
        "\"grammar\": {\"title\": \"Grammar\", \"description\": \"grammar score in the range of 1 to 100\", \"type\": \"integer\"}, "
        "\"relevance\": {\"title\": \"Relevance\", \"description\": \"relevance score in the range of 1 to 100\", \"type\": \"integer\"}, "
        "\"overall_score\": {\"title\": \"Overall Score\", \"description\": \"average score of appropriateness, coherence, content, and grammar\", \"type\": \"integer\"}}, "
        "\"required\": [\"appropriateness\", \"coherence\", \"content\", \"grammar\", \"relevance\", \"overall_score\"]}\n"
        "\n An example output:\n"
        "{\"appropriateness\":75,\"coherence\":66,\"content\":82,\"grammar\":77,\"relevance\":33,\"overall_score\":66}\n"
        "The JSON object must start with '{' and end with '}', without any extra text, newlines, or symbols outside of the object. "
        "Only provide the JSON object in the output.\n"
        "Score the above dialogue response on a continuous scale from 1 to 100.\n\n"
    )

    evaluation_results = []
    evaluation_errors = []

    # Iterate over each dialogue instance.
    for i in range(len(predictions)):
        prompt = (
            f"Context: {contexts[i]}\n"
            f"Dialogues: {dialogues[i]}\n"
            f"Assistant answer: {predictions[i]}\n"
            f"System Prompt: {system_prompt}"
        )
        try:
            # Call the judge model with the constructed prompt.
            evaluation = judge(prompt, max_new_tokens=100, response_format="json")
        except Exception as e:
            print(judge.model_name)
            print(e)
            error_entry = {
                "id": ids[i],
                "model_name": judge.model_name,
                "error": str(e)
            }
            evaluation_errors.append(error_entry)
            continue

        try:
            # Parse the judge's output as JSON.
            result = json.loads(evaluation)
            result["id"] = ids[i]  # Append the unique ID to the result.
            evaluation_results.append(result)
        except Exception as e:
            print("Result could not be parsed into a dictionary. The faulty result is logged in the error log.")
            error_entry = {
                "id": ids[i],
                "model_name": judge.model_name,
                "output": evaluation
            }
            evaluation_errors.append(error_entry)
            continue

    return evaluation_results, evaluation_errors


def compute_dialog(model, all_models, subset_length=10):
    """
    Evaluates a model on a subset of the TopicalChat dataset and saves the predictions along with the dialogue context.

    This function loads a subset of the dataset, uses the model to generate a dialogue continuation, and saves the generated
    responses along with the context and expected dialogue. It then evaluates these responses using judge models.

    Args:
        model (function): A function that calls the model and returns a dialogue continuation.
        all_models (list): A list of judge models for evaluating the dialogue responses.
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        dict: A dictionary containing evaluation results from each judge model.
    """
    # Load the TopicalChat dataset.
    dataset = dl.load_dataset("topical_chat")
    subset = dataset[:subset_length]

    predictions = []
    contexts = []
    dialogues = []
    ids = []

    # Loop through each dataset entry.
    for entry in subset:
        context = entry['context']  # Context of the dialogue.
        dialogue = entry['source']  # Expected dialogue continuation.
        ids.append(entry['id'])

        # Create the input text for the model.
        input_text = (
            f"Context: {context}\n"
            f"Dialog: {dialogue}\n"
            "The dialogue is between two humans. With every line break, the speaker changes. "
            "Generate an appropriate continuation of the dialogue based on the context and the previous dialogue. "
            "Your answer should only contain the dialogue continuation. Use approximately 50-75 words."
        )

        try:
            # Generate the dialogue continuation.
            model_response = model(input_text, max_new_tokens=150)
        except Exception as e:
            print(model.model_name)
            print(e)
            print(entry)
            continue

        contexts.append(context)
        predictions.append(model_response.strip())
        dialogues.append(dialogue)

    # Save the predictions, dialogues, and contexts to a JSON file.
    output_data = {
        "predictions": predictions,
        "dialogues": dialogues,
        "contexts": contexts,
        "ids": ids
    }

    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dialog_predictions_references.json")

    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Predictions and references have been saved to {output_file}.")

    results = {}

    # Evaluate the dialogue responses using each judge model.
    for judge in all_models:
        judge_results, errors = llm_dialog_metric(predictions, dialogues, contexts, ids, judge)
        # Retain only valid evaluation results.
        results[judge.model_name] = [r for r in judge_results if r is not None]
        print(f"Evaluation of {model.model_name} by {judge.model_name} completed")

        # Save any errors encountered during evaluation.
        error_dir = f"Errors/llm_dialog_Errors/Errors_{model.model_name}/{judge.model_name}"
        os.makedirs(error_dir, exist_ok=True)
        error_file = os.path.join(error_dir, "errors.json")
        with open(error_file, "w") as file:
            json.dump(errors, file, indent=4)

    return results


def calculate_average_scores(results):
    """
    Calculates average scores for each metric (e.g., appropriateness, coherence, content, grammar, relevance, overall_score)
    per judge and overall for each model.

    Args:
        results (dict): A dictionary containing evaluation results from various judges for different models.

    Returns:
        dict: A dictionary containing the average scores per metric.
    """
    averaged_results = {}

    for model_name, judges in results.items():
        model_averages = {}
        for judge_name, evaluations in judges.items():
            # Initialize lists for each metric.
            scores = {"appropriateness": [], "coherence": [], "content": [], "grammar": [], "relevance": [],
                      "overall_score": []}

            # Collect scores for each metric from all evaluations.
            for evaluation in evaluations:
                for key in scores.keys():
                    if key in evaluation:
                        scores[key].append(evaluation[key])

            # Compute average score for each metric.
            judge_average = {key: np.mean(values) if values else 0 for key, values in scores.items()}
            model_averages[judge_name] = judge_average

        # Compute overall averages across all judges.
        overall_scores = {"appropriateness": [], "coherence": [], "content": [], "grammar": [], "relevance": [],
                          "overall_score": []}
        for judge_average in model_averages.values():
            for key in overall_scores.keys():
                overall_scores[key].append(judge_average[key])
        overall_average = {key: np.mean(values) if values else 0 for key, values in overall_scores.items()}
        model_averages["overall_average"] = overall_average

        averaged_results[model_name] = model_averages

    return averaged_results


def llm_dialog_eval(models, subset_length=10):
    """
    Evaluates dialogue capabilities for all provided models using LLM-as-a-Judge and saves the aggregated results.

    This function iterates over a list of models, computes dialogue evaluations using the compute_dialog function,
    calculates average scores for each metric, and saves the overall results in a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of dataset entries to use for evaluation.
    """
    all_results = {}
    for model in models:
        result = compute_dialog(model, models, subset_length)
        all_results[model.model_name] = result

    averaged_results = calculate_average_scores(all_results)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), "Results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "llm_dialog_all_models_results.json")

    with open(results_file, "w") as file:
        json.dump({"all_results": all_results, "averaged_results": averaged_results}, file, indent=4)

    print(f"All LLM-EVAL results have been saved to {results_file}.")
