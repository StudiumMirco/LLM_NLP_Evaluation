import os
import json
import numpy as np

from dataload.dataload import DatasetLoader as dl


def llmstory_metric(predictions, references, prompts, constraint_words_list, ids, judge):
    """
    Computes LLM-EVAL metrics for story generation predictions using an LLM judge.

    Args:
        predictions (list): A list of model-generated story continuations.
        references (list): A list of reference story continuations.
        prompts (list): A list of story prompts.
        constraint_words_list (list): A list of constraint words that were required in the story.
        ids (list): A list of unique story identifiers.
        judge (function): The judge LLM model used to evaluate the quality of the generated stories.

    Returns:
        tuple: A tuple containing:
            - evaluation_results (list): A list of evaluation results (each as a dictionary parsed from JSON output).
            - evaluation_errors (list): A list of error logs (each as a dictionary containing error details).
    """
    # Define the system prompt for the judge LLM.
    system_prompt = (
        "Please act as an impartial judge and evaluate the quality of the continuation of a story provided by an AI assistant. "
        "You will be given the story prompt, the assistant's answer, a reference continuation, and a list of constraint words. "
        "Your evaluation should focus on the assistant's answer. Be as objective as possible. "
        "The output should be formatted as a JSON instance that strictly conforms to the following JSON schema:\n"
        "Here is the output schema: {\"properties\": {\"coherence\": {\"title\": \"Coherence\", \"description\": \"coherence score in the range of 1 to 100\", \"type\": \"integer\"},"
        "\"style\": {\"title\": \"Style\", \"description\": \"style score in the range of 1 to 100\", \"type\": \"integer\"},"
        "\"fluency\": {\"title\": \"Fluency\", \"description\": \"fluency score in the range of 1 to 100\", \"type\": \"integer\"},"
        "\"creativity\": {\"title\": \"Creativity\", \"description\": \"creativity score in the range of 1 to 100\", \"type\": \"integer\"},"
        "\"grammar\": {\"title\": \"Grammar\", \"description\": \"grammar score in the range of 1 to 100\", \"type\": \"integer\"},"
        "\"overall_score\": {\"title\": \"Overall Score\", \"description\": \"average score of coherence, fluency, style, creativity, and grammar\", \"type\": \"integer\"}},"
        "\"required\": [\"coherence\", \"style\", \"fluency\", \"creativity\", \"grammar\", \"overall_score\"]}\n"
        "\n An example output:\n"
        "{\"coherence\":75,\"style\":82,\"fluency\":77,\"creativity\":33,\"grammar\":45,\"overall_score\":62}\n"
        "The JSON object must start with '{' and end with '}', without any extra text, newlines, or symbols outside of the object. "
        "Only provide the JSON object in the output.\n"
        "1. Read the story prompt carefully and understand the main context and storyline. \n"
        "2. Read the assistant's continuation and compare it to the story prompt, the reference continuation, and consider the use of constraint words. "
        "Check if the continuation is coherent, creative, fluent, follows grammatical rules, and matches the writing style of the prompt, "
        "while also appropriately incorporating all constraint words. \n"
        "3. Assign scores on a continuous scale of 1 to 100.\n\n"
    )

    evaluation_results = []
    evaluation_errors = []
    # Iterate over each generated story continuation.
    for i in range(len(predictions)):
        # Construct the evaluation prompt by combining the system prompt with story-specific data.
        prompt = (
            f"System Prompt: {system_prompt}"
            f"Story Prompt: {prompts[i]}\n"
            f"Assistant answer: {predictions[i]}\n"
            f"Reference: {references[i]}\n"
            f"Constraint Words: {constraint_words_list[i]}\n"
        )
        try:
            # Call the judge model with the evaluation prompt.
            evaluation = judge(prompt, max_new_tokens=100, response_format="json")
        except Exception as e:
            print(judge.model_name)
            print(e)
            evaluation_errors.append({
                "story_id": ids[i],
                "judge_name": judge.model_name,
                "error": str(e)
            })
            continue

        try:
            # Parse the judge's output as JSON.
            result = json.loads(evaluation)
            result["story_id"] = ids[i]
            evaluation_results.append(result)
        except Exception as e:
            print("Result could not be parsed into a dictionary. The faulty result is logged in the error log.")
            evaluation_errors.append({
                "story_id": ids[i],
                "judge_name": judge.model_name,
                "error": str(e),
                "Evaluation": evaluation
            })
            continue

    return evaluation_results, evaluation_errors


def compute_llmstory(model, all_models, subset_length=10):
    """
    Evaluates a model on a subset of the ROCStories dataset for story continuation and computes evaluation metrics using judge LLMs.

    This function loads a subset of the ROCStories dataset, uses the provided model to generate story continuations,
    saves the generated continuations along with reference continuations and additional metadata, and then evaluates
    the predictions using each judge model.

    Args:
        model (function): A function that calls the model and returns a story continuation for the given prompt.
        all_models (list): A list of judge models that will evaluate the generated story continuations.
        subset_length (int): The number of dataset entries to use for evaluation.

    Returns:
        dict: A dictionary where each key is a judge model's name and each value is a list of evaluation results.
    """
    # Load the ROCStories dataset.
    dataset = dl.load_dataset("rocstories")
    subset = dataset[:subset_length]

    # Initialize lists to store story prompts, predictions, references, constraint words, and story IDs.
    predictions = []
    references = []
    prompts = []
    ids = []
    constraint_words_list = []

    # Iterate over the selected dataset entries.
    for entry in subset:
        prompt = entry['prompt']
        reference_continuation = entry['continuation']
        constraint_words = entry['constraint_words']
        story_id = entry['story_id']

        ids.append(story_id)
        prompts.append(prompt)
        constraint_words_list.append(constraint_words)

        # Create the input text for the summarization model.
        input_text = (
            "There are certain predefined words that must be included in the continuation of the story. "
            "Use all the constraint words listed below to complete the story in a meaningful and engaging way. "
            "Here is the beginning of the story, followed by the constraint words. Please write a continuation "
            "that includes all the given words and fits well with the story.\n\n"
            f"Story Beginning: \"{prompt}\"\n\n"
            f"Constraint Words: {constraint_words}\n\n"
            "Task: Write a continuation of the story that naturally and coherently builds on the story beginning. "
            "The continuation should meet the following requirements:\n"
            "1. All constraint words must be used in the text.\n"
            "2. The continuation style should match the beginning of the story.\n"
            "3. The story should form an interesting and self-contained narrative.\n"
            "4. The story should have around 100-150 words.\n\n"
            "Continuation of the Story:"
        )

        try:
            # Generate the story continuation using the model.
            response = model(input_text, max_new_tokens=300)
        except Exception as e:
            print(model.model_name)
            print(e)
            print(entry)
            continue

        predictions.append(response.strip())
        references.append(reference_continuation.strip())

    # Save the predictions, references, prompts, constraint words, and story IDs to a JSON file.
    output_data = {
        "story_id": ids,
        "prompts": prompts,
        "constraint_words": constraint_words_list,
        "predictions": predictions,
        "references": references
    }
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "llmstory_predictions_references.json")
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)
    print(f"Predictions and references have been saved to {output_file}.")

    results = {}
    # Evaluate the generated story continuations using each judge model.
    for judge in all_models:
        judge_results, errors = llmstory_metric(predictions, references, prompts, constraint_words_list, ids, judge)
        # Only keep valid evaluation results.
        results[judge.model_name] = [r for r in judge_results if r is not None]
        print(f"Evaluation of {model.model_name} by {judge.model_name} completed")

        # Save evaluation errors to a separate JSON file.
        error_dir = f"Errors/llm_story_Errors/Errors_{model.model_name}/{judge.model_name}"
        os.makedirs(error_dir, exist_ok=True)
        error_file = os.path.join(error_dir, "errors.json")
        with open(error_file, "w") as file:
            json.dump(errors, file, indent=4)

    return results


def calculate_average_scores(results):
    """
    Calculates average evaluation scores for each metric (e.g., coherence, style, etc.) both per judge and overall.

    Args:
        results (dict): A dictionary with evaluation results from various judges for different models.

    Returns:
        dict: A dictionary with the average scores per metric.
    """
    averaged_results = {}

    for model_name, judges in results.items():
        model_averages = {}
        for judge_name, evaluations in judges.items():
            # Initialize lists to collect scores for each metric.
            scores = {"coherence": [], "style": [], "fluency": [], "creativity": [], "grammar": [], "overall_score": []}

            # Collect scores for each metric from the evaluations.
            for evaluation in evaluations:
                for key in scores.keys():
                    if key in evaluation:
                        scores[key].append(evaluation[key])

            # Compute the average score for each metric.
            judge_average = {key: np.mean(values) if values else 0 for key, values in scores.items()}
            model_averages[judge_name] = judge_average

        # Compute overall averages across all judges for the model.
        overall_scores = {"coherence": [], "style": [], "fluency": [], "creativity": [], "grammar": [],
                          "overall_score": []}
        for judge_average in model_averages.values():
            for key in overall_scores.keys():
                overall_scores[key].append(judge_average[key])

        overall_average = {key: np.mean(values) if values else 0 for key, values in overall_scores.items()}
        model_averages["overall_average"] = overall_average

        averaged_results[model_name] = model_averages

    return averaged_results


def llm_story_eval(models, subset_length=10):
    """
    Evaluates story generation quality for all provided models and saves the aggregated results to a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of dataset entries to use for evaluation.
    """
    all_results = {}
    for model in models:
        result = compute_llmstory(model, models, subset_length)
        all_results[model.model_name] = result

    # Calculate average scores per model.
    averaged_results = calculate_average_scores(all_results)

    # Save the aggregated results and averaged scores to a JSON file.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), "Results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "llmstory_all_models_results.json")

    with open(results_file, "w") as file:
        json.dump({"all_results": all_results, "averaged_results": averaged_results}, file, indent=4)

    print(f"All story generation evaluation results have been saved to {results_file}.")
