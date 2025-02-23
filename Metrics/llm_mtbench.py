import os
import json
import numpy as np

from dataload.dataload import DatasetLoader as dl


def llm_writing_metric(predictions, questions, ids, judge):
    """
    Computes LLM-as-a-Judge metrics for writing tasks using an LLM judge (e.g., GPT-4o).

    This function constructs two different prompts (system_prompt and system_prompt2) to evaluate
    the quality of responses for a writing task. The first prompt asks for a short explanation and
    a simple rating, while the second prompt requires a detailed JSON-formatted evaluation based on
    multiple criteria such as coherence, style, content, grammar, relevance, and fluency.

    Args:
        predictions (list): A list where each element is a list of two strings:
                            [assistant's initial response, assistant's follow-up response].
        questions (list): A list where each element is a list of two strings:
                          [initial writing task question, follow-up question].
        ids (list): A list of unique identifiers for each writing task.
        judge (function): The LLM model used as a judge to evaluate the responses.

    Returns:
        tuple: A tuple containing two lists:
            - evaluation_results from the first system prompt (raw text output).
            - evaluation_results2 from the second system prompt (JSON-parsed output),
          along with a list of any evaluation errors.
    """
    # Define the first system prompt for a basic evaluation.
    system_prompt = (
        "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant "
        "to the user question displayed below. There will be one start question and afterwards a follow-up question. "
        "Rate both. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, "
        "and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. "
        "After providing your explanation, please rate the response on a scale of 1 to 100 by strictly following this format: "
        "[[rating]], for example: Rating: [[5]]."
    )

    # Define the second system prompt for a detailed evaluation using a JSON schema.
    system_prompt2 = (
        "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to a writing task. "
        "You will be given a writing task and the assistant's response. Furthermore, you will be given a follow-up question and "
        "the assistant's response to this follow-up question. Evaluate the complete writing task with both assistant responses. "
        "Your evaluation should consider factors such as coherence, style, content, grammar, relevance, and fluency. Be as objective as possible. "
        "The output should be formatted as a JSON instance that strictly conforms to the following JSON schema:\n"
        "{\"properties\": {\"coherence\": {\"title\": \"Coherence\", \"description\": \"coherence score in the range of 1 to 100\", \"type\": \"integer\"},"
        "\"style\": {\"title\": \"Style\", \"description\": \"style score in the range of 1 to 100\", \"type\": \"integer\"},"
        "\"content\": {\"title\": \"Content\", \"description\": \"content score in the range of 1 to 100\", \"type\": \"integer\"},"
        "\"grammar\": {\"title\": \"Grammar\", \"description\": \"grammar score in the range of 1 to 100\", \"type\": \"integer\"},"
        "\"relevance\": {\"title\": \"Relevance\", \"description\": \"relevance score in the range of 1 to 100\", \"type\": \"integer\"},"
        "\"fluency\": {\"title\": \"Fluency\", \"description\": \"fluency score in the range of 1 to 100\", \"type\": \"integer\"},"
        "\"overall_score\": {\"title\": \"Overall Score\", \"description\": \"average score of coherence, style, content, grammar, relevance, and fluency\", \"type\": \"integer\"}},"
        "\"required\": [\"coherence\", \"style\", \"content\", \"grammar\", \"relevance\", \"overall_score\"]}\n"
        "\n An example output:\n"
        "{\"coherence\":75,\"style\":82,\"content\":77,\"grammar\":33,\"relevance\":45,\"fluency\":62,\"overall_score\":62}\n"
        "The JSON object must start with '{' and end with '}', without any extra text, newlines, or symbols outside of the object. "
        "Only provide the JSON object in the output.\n"
        "Score the following writing task response generated on a continuous scale from 1 to 100.\n\n"
    )

    evaluation_results = []
    evaluation_errors = []

    # First evaluation pass using system_prompt (raw text response).
    for i in range(len(predictions)):
        prompt = (
            f"System Prompt: {system_prompt}"
            f"Writing Task: {questions[i][0]}\n"
            f"Assistant's Response: {predictions[i][0]}\n"
            f"User Follow-up question: {questions[i][1]}\n"
            f"Assistant's Response to follow-up: {predictions[i][1]}\n"
        )
        try:
            evaluation = judge(prompt, max_new_tokens=250)
        except Exception as e:
            print(judge.model_name)
            print(e)
            error_entry = {
                "question_id": ids[i],
                "model": judge.model_name,
                "error": str(e),
            }
            evaluation_errors.append(error_entry)
            continue
        result_text = {"id": ids[i], "result": evaluation}
        evaluation_results.append(result_text)

    evaluation_results2 = []

    # Second evaluation pass using system_prompt2 (expects JSON output).
    for i in range(len(predictions)):
        prompt = (
            f"System Prompt: {system_prompt2}"
            f"Writing Task: {questions[i][0]}\n"
            f"Assistant's Response: {predictions[i][0]}\n"
            f"User Follow-up question: {questions[i][1]}\n"
            f"Assistant's Response to follow-up: {predictions[i][1]}\n"
        )
        try:
            evaluation = judge(prompt, max_new_tokens=250, response_format="json")
        except Exception as e:
            print(judge.model_name)
            print(e)
            error_entry = {
                "question_id": ids[i],
                "model": judge.model_name,
                "error": str(e),
            }
            evaluation_errors.append(error_entry)
            continue
        try:
            result = json.loads(evaluation)
            result["question_id"] = ids[i]  # Add question_id to the result
            evaluation_results2.append(result)
        except Exception as e:
            print("Result could not be parsed into a dictionary. The faulty result is logged in the error log.")
            error_entry = {
                "question_id": ids[i],
                "model": judge.model_name,
                "error": str(e),
                "output": evaluation if 'evaluation' in locals() else "N/A"
            }
            evaluation_errors.append(error_entry)
            continue

    return evaluation_results, evaluation_results2, evaluation_errors


def compute_writing(model, all_models, subset_length=10):
    """
    Evaluates a model on a subset of the mtbench dataset for writing tasks and saves the predictions.

    This function loads the mtbench dataset, uses the model to generate an initial response (Turn 1)
    and a follow-up response (Turn 2) for a given writing task, and then stores these responses along
    with the corresponding questions and IDs.

    Args:
        model (function): A function that calls the model and returns a response for a given text.
        all_models (list): A list of models that will be used as judges for evaluation.
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        tuple: A tuple containing two dictionaries:
            - The first dictionary contains the raw evaluation results from each judge.
            - The second dictionary contains the detailed evaluation results from each judge.
    """
    # Load the mtbench dataset.
    dataset = dl.load_dataset("mtbench")
    subset = dataset[:subset_length]

    predictions = []
    questions = []
    ids = []

    # Iterate over each entry in the dataset subset.
    for entry in subset:
        # The writing task (Turn 1) is the initial question.
        question = entry['turns'][0]
        input_text = f"Writing Task: {question}\n"
        try:
            # Generate the assistant's initial response.
            model_response = model(input_text, max_new_tokens=500)
        except Exception as e:
            print(model.model_name)
            print(e)
            print(entry)
            continue

        # Prepare a multi-turn input for the follow-up response (Turn 2).
        messages = [
            {"role": "user", "content": entry["turns"][0]},
            {"role": "assistant", "content": model_response},
            {"role": "user", "content": entry["turns"][1]}
        ]
        try:
            model_response_turn_2 = model.predict_multi_turn(messages, max_new_tokens=500)
        except Exception as e:
            print(model.model_name)
            print(e)
            print(entry)
            continue

        questions.append(entry["turns"])
        ids.append(entry['question_id'])
        predictions.append([model_response.strip(), model_response_turn_2])

    # Save the writing task predictions and questions to a JSON file.
    output_data = {
        "ids": ids,
        "predictions": predictions,
        "questions": questions,
    }
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "writing_predictions.json")
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)
    print(f"Predictions have been saved to {output_file}.")

    # Evaluate the writing responses using each judge model.
    results = {}
    results_written = {}
    for judge in all_models:
        # The llm_writing_metric returns two sets of results: one from the first prompt and one from the second.
        results_written[judge.model_name], results[judge.model_name], errors = llm_writing_metric(predictions,
                                                                                                  questions, ids, judge)
        print(f"Evaluation of {model.model_name} by {judge.model_name} completed")

        # Save any errors encountered during evaluation.
        error_dir = f"Errors/llm_mtbench_Errors/Errors_{model.model_name}/{judge.model_name}"
        os.makedirs(error_dir, exist_ok=True)
        error_file = os.path.join(error_dir, "errors.json")
        with open(error_file, "w") as file:
            json.dump(errors, file, indent=4)

    return results_written, results


def calculate_average_scores(results):
    """
    Calculates the average scores for each metric (e.g., coherence, style, content, grammar, relevance, fluency, overall_score)
    for each judge and overall for each model.

    Args:
        results (dict): A dictionary containing evaluation results from various judges for different models.

    Returns:
        dict: A dictionary containing the average scores per metric.
    """
    averaged_results = {}

    for model_name, judges in results.items():
        model_averages = {}
        for judge_name, evaluations in judges.items():
            scores = {"coherence": [], "style": [], "content": [], "grammar": [], "relevance": [], "fluency": [],
                      "overall_score": []}
            for evaluation in evaluations:
                if isinstance(evaluation, dict):
                    for key in scores.keys():
                        if key in evaluation:
                            scores[key].append(evaluation[key])
            judge_average = {key: np.mean(values) if values else 0 for key, values in scores.items()}
            model_averages[judge_name] = judge_average

        overall_scores = {"coherence": [], "style": [], "content": [], "grammar": [], "relevance": [], "fluency": [],
                          "overall_score": []}
        for judge_average in model_averages.values():
            for key in overall_scores.keys():
                overall_scores[key].append(judge_average[key])
        overall_average = {key: np.mean(values) if values else 0 for key, values in overall_scores.items()}
        model_averages["overall_average"] = overall_average

        averaged_results[model_name] = model_averages

    return averaged_results


def llm_writing_eval(models):
    """
    Evaluates writing task performance for all provided models using LLM-as-a-Judge and saves the results.

    This function iterates over a list of models, computes writing task evaluations using the compute_writing
    function, calculates average scores, and saves the aggregated results in a JSON file.

    Args:
        models (list): A list of models to be evaluated.
    """
    all_results = {}
    all_results_written = {}
    for model in models:
        result_written, result = compute_writing(model, models)
        all_results[model.model_name] = result
        all_results_written[model.model_name] = result_written

    averaged_results = calculate_average_scores(all_results)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "llm_writing_all_models_results.json")

    with open(results_file, "w") as file:
        json.dump(
            {
                "all_results": all_results,
                "averaged_results": averaged_results,
                "all_results_written": all_results_written
            },
            file,
            indent=4
        )

    print(f"All LLM-EVAL results have been saved to {results_file}.")
