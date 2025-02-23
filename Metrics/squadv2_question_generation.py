import os
import json


from Metrics.rouge import rouge_metric
from Models.models import OpenAIModel
from dataload.dataload import DatasetLoader as dl


def calculate_accuracy(y_true, y_pred):
    # Calculate accuracy: the number of correct predictions divided by the total number of predictions.
    correct = sum([1 for true, pred in zip(y_true, y_pred) if pred in true])
    accuracy = correct / len(y_true)
    return accuracy


def qg_metric(questions):
    """
    Evaluates the quality of generated questions (QG metric) using a GPT-4o instance.

    This function compares generated questions with reference questions from the SQuADv2 dataset.

    Args:
        questions (list): A list of generated questions (predictions).

    Returns:
        dict: A dictionary containing evaluation results, including individual results and accuracy metrics.
    """
    # Instantiate a GPT-4o model from OpenAI.
    gpt4o = OpenAIModel(model_name="gpt-4o", max_new_tokens=1000, temperature=0, system_prompt=None)

    # Initialize lists to store results for reference and predicted questions.
    ref_results = []  # Results for reference questions.
    pred_results = []  # Results for predicted questions.
    # Initialize a dictionary to count different outcome classes.
    class_counts = {
        "ReferenceTruePredictionTrue": 0,
        "ReferenceTruePredictionFalse": 0,
        "ReferenceFalsePredictionTrue": 0,
        "ReferenceFalsePredictionFalse": 0
    }

    # Load the SQuADv2 dataset.
    dataset = dl.load_dataset("squad_v2")  # Load SQuADv2 dataset.

    # (Optional check for equal list lengths is omitted.)

    # Define an inner function to evaluate an answer given a question, context, and correct answers.
    def evaluate_answer(question, context, correct_answers):
        # Construct the input prompt for GPT-4o.
        input_text = (
            f"Context: {context}\n"
            f"Based on this context, answer the following question precisely:\n"
            f"Question: {question}\n"
            f"Answer only with the exact information from the context. If there is no suitable answer, respond with: \"No answer possible\".\n"
            f"Do not explain your answer!"
        )

        try:
            # Generate a response, keeping only alphanumeric characters and whitespace.
            response = ''.join(
                char for char in gpt4o(input_text, max_new_tokens=50) if char.isalnum() or char.isspace()).strip()
            # If the response indicates no answer, convert it to an empty string.
            if response in ["No answer possible", "No answer possible.", "No answer possible\n"]:
                response = ""
            # Check if the cleaned response matches any of the correct answers (ignoring case and non-alphanumeric characters).
            is_correct = any(
                response.lower() == ''.join(
                    char for char in correct_answer if char.isalnum() or char.isspace()).strip().lower()
                for correct_answer in correct_answers
            )
            return is_correct, response
        except Exception as e:
            print(gpt4o.model_name)
            print(e)
            return False, "Error"

    # Iterate over the dataset entries, matching the number of questions provided.
    for idx, entry in enumerate(dataset[:len(questions)]):
        # Extract the reference question, context, and answer from the dataset entry.
        ref_question = entry['question']
        ref_context = entry['context']
        ref_answer = entry['answers']['text'] if entry['answers']['text'] else [""]

        # Evaluate the reference question.
        ref_correct, ref_response = evaluate_answer(ref_question, ref_context, ref_answer)
        ref_results.append({
            "question": ref_question,
            "response": ref_response,
            "correct_answer": ref_answer,
            "correct": ref_correct
        })

        # Evaluate the predicted question.
        pred_question = questions[idx]
        pred_correct, pred_response = evaluate_answer(pred_question, ref_context, ref_answer)
        pred_results.append({
            "question": pred_question,
            "response": pred_response,
            "correct_answer": ref_answer,
            "correct": pred_correct
        })

        # Update classification counts based on the evaluation outcomes.
        if ref_correct and pred_correct:
            class_counts["ReferenceTruePredictionTrue"] += 1
        elif ref_correct and not pred_correct:
            class_counts["ReferenceTruePredictionFalse"] += 1
        elif not ref_correct and pred_correct:
            class_counts["ReferenceFalsePredictionTrue"] += 1
        else:
            class_counts["ReferenceFalsePredictionFalse"] += 1

    # Calculate overall accuracy for both reference and predicted questions.
    ref_accuracy = sum(1 for r in ref_results if r["correct"]) / len(ref_results) if ref_results else 0
    pred_accuracy = sum(1 for r in pred_results if r["correct"]) / len(pred_results) if pred_results else 0

    # Prepare the output data containing all evaluation results.
    output_data = {
        "ReferenceResults": ref_results,
        "PredictionResults": pred_results,
        "ClassCounts": class_counts,
        "ReferenceAccuracy": ref_accuracy,
        "PredictionAccuracy": pred_accuracy
    }

    # (Optional) Code to save the results to a JSON file is commented out.
    # output_dir = "test/qg_metric_results"
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, "qg_metric_results.json")
    # with open(output_file, "w") as file:
    #     json.dump(output_data, file, indent=4)
    # print(f"Results have been saved to {output_file}.")

    # Return the evaluation results.
    return output_data


# def qg_metric_alt(questions, answers, contexts):
#     """
#     Alternative QG metric function that evaluates generated questions using a GPT-4o instance.
#
#     This function processes lists of questions, corresponding reference answers, and contexts,
#     and computes the accuracy of the model's responses.
#
#     Args:
#         questions (list): A list of generated questions.
#         answers (list): A list of reference answers.
#         contexts (list): A list of contexts for the questions.
#
#     Returns:
#         dict: A dictionary containing the computed accuracy (and optionally F1 score).
#     """
#     # Instantiate a GPT-4o model.
#     gpt4o = OpenAIModel(model_name="gpt-4o", max_new_tokens=1000, temperature=0, system_prompt=None)
#
#     gpt_responses = []
#     if len(questions) != len(contexts):
#         print("The lists must be of equal length.")
#     else:
#         for i in range(len(questions)):
#             question = questions[i]
#             context = contexts[i]
#             # Construct the input prompt for the API.
#             input_text = (
#                 f"Context: {context}\n"
#                 f"Based on this context, answer the following question precisely:\n"
#                 f"Question: {question}\n"
#                 f"Answer only with the exact information from the context. If there is no suitable answer, respond with: \"No answer possible\""
#                 f" Do not explain your answer!"
#             )
#
#             try:
#                 response = gpt4o(input_text, max_new_tokens=50)
#             except Exception as e:
#                 print(gpt4o.model_name)
#                 print(e)
#                 continue
#
#             # If the response indicates no answer, set it to an empty string.
#             if response in ["No answer possible", "No answer possible."]:
#                 response = ""
#             print(response)
#             gpt_responses.append(response)
#
#     # Calculate the accuracy of the generated responses.
#     accuracy = calculate_accuracy(answers, gpt_responses)
#     # Calculation of F1 score is commented out.
#     # f1 = calculate_f1_score(answers, gpt_responses)
#
#     print("Accuracy:", accuracy)
#     # print("F1 Score:", f1)
#
#     return {
#         "accuracy": accuracy,
#         # "f1_score": f1
#     }


def compute_squad_qg(model, subset_length=10):
    """
    Evaluates a model on a subset of the SQuADv2 dataset by generating questions for a given answer
    and saves both the predictions and reference questions to a file.

    Args:
        model (function): A function that calls the model and returns a generated question for the given context.
        subset_length (int): The number of data samples from the SQuADv2 dataset to use for evaluation.

    Returns:
        dict: A dictionary containing computed metrics including Rouge scores and QG metrics.
    """
    # Load the SQuADv2 dataset.
    dataset = dl.load_dataset("squad_v2")  # Load SQuADv2 dataset.
    subset = dataset[:subset_length]

    # Initialize lists to store predictions, reference questions, answers, and contexts.
    predictions = []
    references = []
    answers = []
    contexts = []

    # Iterate over the subset of the dataset.
    for entry in subset:
        context = entry['context']
        question = entry['question']
        answer = entry['answers']['text'][0]  # Take the first reference answer.

        # Construct the input prompt for question generation.
        input_text = (
            f"Context: {context}\n  Answer: {answer}\n"
            f"Generate the most probable Question for the given answer based on the context provided."
            f"If the answer is empty, generate a question that cannot be answered with the information given in the context."
            f"Your answer should only contain the question"
        )

        try:
            # Call the model with the constructed prompt.
            response = model(input_text, max_new_tokens=50)
        except Exception as e:
            # On error, print details and record an error for this entry.
            print(model.model_name)
            print(e)
            print(entry)
            predictions.append("Error")
            references.append(question)
            answers.append(answer)
            contexts.append(context)
            continue

        # If answers exist, assign the list of answers; otherwise, default to an empty list.
        if entry['answers']['text']:
            answer = entry['answers']['text']  # List of reference answers.
        else:
            answer = [""]

        # Save the model's prediction and the corresponding reference.
        predictions.append(response.strip())
        references.append(question)
        answers.append(answer)
        contexts.append(context)

    # Prepare output data containing predictions and reference questions.
    output_data = {
        "predictions": predictions,
        "references": references
    }

    # Create a dynamic directory path for saving results specific to the model.
    output_dir = f"PredictionOutputs/PredictionOutputs_{model.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Define the output filename.
    output_file = os.path.join(output_dir, "squad_qg_predictions_references.json")

    # Write the output data to a JSON file.
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Predictions and references have been saved to {output_file}.")

    results = {}
    # Compute the Rouge metric between predictions and reference questions.
    results["Rouge"] = rouge_metric(predictions, references)
    # Compute the QG metric using the qg_metric function.
    results["QG-Metrik"] = qg_metric(predictions)

    # Create a dynamic path for saving detailed QG metric results.
    output_file = os.path.join(output_dir, "squad_qgmetric_detailed_results.json")

    # Write the detailed QG metric results to a JSON file.
    with open(output_file, "w") as file:
        json.dump(results["QG-Metrik"], file, indent=4)

    print(f"Detailed QG metric results have been saved to {output_file}.")

    # Compute the average Rouge score and normalize it.
    rouge = results["Rouge"]
    avg_rouge = ((rouge["rouge1"] + rouge["rouge2"] + rouge["rougeL"] + rouge["rougeLsum"]) / 4) * 100
    # Normalize the QG metric (assuming 'PredictionResults' is a numerical value).
    qg_metric_norm = results["QG-Metrik"]["PredictionResults"] * 100

    # Combine the average Rouge score and normalized QG metric equally to form an overall score.
    # Not used anymore, because ROUGE has not shown good results in evaluating QG
    # overall_score = (avg_rouge + qg_metric_norm) / 2

    results["Final_scores"] = {
        "Average_Rouge": avg_rouge,
        "QG_metric_norm": qg_metric_norm,
        # Rouge is not used for the Overall_score anymore
        "Overall_Score": qg_metric_norm
    }
    return results


def squad_eval_qg(models, subset_length=10):
    """
    Performs SQuADv2 evaluation for all provided models and saves the overall results to a JSON file.

    Args:
        models (list): A list of models to be evaluated.
        subset_length (int): The number of data samples from the SQuADv2 dataset to use for each model.
    """
    all_results = {}
    for model in models:
        result = compute_squad_qg(model, subset_length)
        all_results[model.model_name] = result

    # Create a directory for final results.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(cur_dir), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    # Define the output filename for overall SQuADv2 results.
    results_file = os.path.join(results_dir, "squad_qg__all_models_results.json")

    # Write all results to a JSON file.
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4)

    print(f"All SQuADv2 results have been saved to {results_file}.")
