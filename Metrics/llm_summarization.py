import os
import json
from dataload.dataload import DatasetLoader as dl


# Define the LLM summary evaluation metric function using GPT-4 as a judge.
def llm_sum_metric(predictions, references, articles, ids, judge):
    """
    Computes LLM-EVAL metrics for the provided model predictions using a judge LLM (e.g., GPT-4o).

    This function uses a detailed system prompt to instruct the judge LLM to evaluate the quality of
    the summaries by comparing the assistant's answer with a reference summary. The evaluation is expected
    to follow a strict JSON schema.

    Args:
        predictions (list): A list of generated summaries from the model.
        references (list): A list of reference summaries.
        articles (list): A list of the original articles that were summarized.
        ids (list): A list of unique identifiers corresponding to each summary.
        judge (function): A judge LLM model (e.g., GPT-4o) that will evaluate the summaries.

    Returns:
        tuple: A tuple containing:
            - evaluation_results (list): A list of evaluation results (each is a dictionary parsed from JSON output).
            - evaluation_errors (list): A list of error logs (each is a dictionary with error details).
    """

    # Define the system prompt that instructs the judge on how to evaluate the summaries.
    system_prompt = (
        "Please act as an impartial judge and evaluate the quality of the summary of an article provided by an AI assistant. "
        "You got the article, the assistant's answer and a reference summary. "
        "Your evaluation should focus on the assistant's answer. "
        "Be as objective as possible. "
        "The output should be formatted as a JSON instance that strictly conforms to the following JSON schema :\n"
        "{\"properties\": {\"coherence\": {\"title\": \"Coherence\", \"description\": \"coherence score in the range of 1 to 100\", \"type\": \"integer\"},"
        "\"consistency\": {\"title\": \"Consistency\", \"description\": \"consistency score in the range of 1 to 100\", \"type\": \"integer\"},"
        "\"grammar\": {\"title\": \"Grammar\", \"description\": \"grammar score in the range of 1 to 100\", \"type\": \"integer\"},"
        "\"relevance\": {\"title\": \"Relevance\", \"description\": \"relevance score in the range of 0 to 100\", \"type\": \"integer\"},"
        "\"fluency\": {\"title\": \"Fluency\", \"description\": \"fluency score in the range of 1 to 100\", \"type\": \"integer\"},"
        "\"overall_score\": {\"title\": \"Overall Score\", \"description\": \"average score of coherence, fluency, style, creativity, and grammar\", \"type\": \"integer\"}},"
        "\"required\": [\"coherence\", \"consistency\", \"grammar\", \"relevance\", \"fluency\", \"overall_score\"]}\n"
        "\n An example output:\n"
        "{\"coherence\":75,\"consistency\":82,\"grammar\":77,\"relevance\":33,\"fluency\":45,\"overall_score\":62}\n"
        "The JSON object must start with '{' and end with '}', without any extra text, newlines, or symbols outside of the object. "
        "Only provide the JSON object in the output.\n"
        "1. Read the news article carefully and identify the main topic and key points. \n"
        "2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, "
        "and if it presents them in a clear and logical order, while avoiding superfluous details.\n"
        "   Check if the summary is coherent, consistent, fluent and follows grammatical rules,\n"
        "3. Assign scores on a continuous scale of 1 to 100.\n\n"
    )

    # Initialize lists to hold the evaluation results and any errors encountered.
    evaluation_results = []
    evaluation_errors = []

    # Iterate over each prediction in the provided list.
    for i in range(len(predictions)):
        # Construct the prompt for the judge by including the original article, the model's summary, the reference summary,
        # and the system prompt with evaluation instructions.
        prompt = (
            f"Article: {articles[i]}\n"
            f"Assistant answer: {predictions[i]}\n"
            f"Reference: {references[i]}\n"
            f"System Prompt: {system_prompt}"
        )

        try:
            # Call the judge model with the constructed prompt.
            # The judge is expected to return a JSON-formatted response.
            evaluation = judge(prompt, max_new_tokens=100, response_format="json")
        except Exception as e:
            # If there is an error during the model call, log the error details.
            print(judge.model_name)
            print(e)
            evaluation_errors.append({
                "id": ids[i],
                "judge_name": judge.model_name,
                "error": str(e)
            })
            continue

        try:
            # Attempt to parse the judge's output as JSON.
            result = json.loads(evaluation)
            # Add the unique identifier to the result.
            result["id"] = ids[i]
            evaluation_results.append(result)
        except Exception as e:
            # If parsing fails, log the error along with the raw output.
            print("Result could not be parsed into a dictionary. The faulty result is logged in the error log.")
            evaluation_errors.append({
                "id": ids[i],
                "error": str(e),
                "output": evaluation
            })
            continue

    # Return both the evaluation results and any errors encountered.
    return evaluation_results, evaluation_errors


# Extended summarization evaluation method using LLM-EVAL (LLM as Judge).
def compute_llsum(model, all_models, subset_length=10):
    """
    Evaluates a model on a subset of the CNN/DailyMail dataset for summarization, and then uses various judge models
    to assess the quality of the generated summaries.

    This function loads a subset of the CNN/DailyMail dataset, uses the provided summarization model to generate summaries,
    saves the predictions along with the reference summaries, and then iterates through the list of judge models to compute
    evaluation metrics.

    Args:
        model (function): A function that calls the summarization model and returns a summary for the given text.
        all_models (list): A list of judge models that will evaluate the generated summaries.
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        dict: A dictionary where each key is a judge model's name and each value is the list of evaluation results from that judge.
    """

    # Load the CNN/DailyMail dataset.
    dataset = dl.load_dataset("cnn_dailymail")  # Instantiate the CNN/DailyMail dataset loader.
    subset = dataset[:subset_length]

    # Initialize lists to store generated summaries, reference summaries, original articles, and unique IDs.
    predictions = []
    references = []
    articles = []
    ids = []

    # Loop over the selected dataset subset.
    for entry in subset:
        article = entry['article']  # The original article text.
        reference_summary = entry['highlights']  # The reference summary.
        ids.append(entry['id'])  # Append the unique ID of the entry.

        # Create the input text for the summarization model.
        input_text = f"Summarize the following article:{article}. Your answer should not be longer than 50 words."

        try:
            # Call the summarization model with the input text.
            response = model(input_text, max_new_tokens=75)
        except Exception as e:
            # If an error occurs during model inference, log the error and skip this entry.
            print(model.model_name)
            print(e)
            print(entry)
            continue

        # Append the generated summary and the original article to their respective lists.
        predictions.append(response.strip())
        articles.append(article)
        # Append the reference summary.
        references.append(reference_summary.strip())

    # Save the predictions, references, and IDs to a JSON file.
    output_data = {
        "id": ids,
        "predictions": predictions,
        "references": references
    }
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "llsum_predictions_references.json")
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)
    print(f"Predictions and references have been saved to {output_file}.")

    # Initialize a dictionary to hold evaluation results from each judge model.
    results = {}

    # Iterate over each judge model in the provided list.
    for judge in all_models:
        # Compute the evaluation metrics for the generated summaries using the current judge model.
        eval_results, eval_errors = llm_sum_metric(predictions, references, articles, ids, judge)
        print(f"Evaluation of {model.model_name} by {judge.model_name} completed")
        results[judge.model_name] = eval_results

        # Save any evaluation errors to a separate JSON file.
        error_dir = f"llm_sum_Errors/Errors_{model.model_name}"
        os.makedirs(error_dir, exist_ok=True)
        error_file = os.path.join(error_dir, f"{judge.model_name}_errors.json")
        with open(error_file, "w") as file:
            json.dump(eval_errors, file, indent=4)
        # Note: The following return statement is inside the loop, meaning it will return after the first judge.
        return results

    return results


def llm_sum_eval(models, subset_length=10):
    """
    Evaluates the summarization quality for all provided models using a LLM-as-a-Judge-Metric, and saves the aggregated results.

    This function iterates over a list of summarization models, evaluates each one using the compute_llsum function,
    and then saves the aggregated evaluation results for all models into a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        None
    """
    all_results = {}
    for model in models:
        result = compute_llsum(model, models, subset_length)
        all_results[model.model_name] = result

    # Create a directory for the final aggregated results.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "llsum_all_models_results.json")

    # Write the aggregated results to a JSON file.
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All summarization evaluation results have been saved to {results_file}.")
