from utils import load_json, save_json
from collections import defaultdict


def nlp_benchmark():
    """
    Aggregates evaluation results from various NLP tasks stored in JSON files and computes
    a final benchmark score for each model. The function processes results from machine translation,
    summarization, dialogue, question answering, question generation, and NLU tasks.

    It reads the task-specific results from JSON files, extracts key scores (such as final overall scores),
    applies weighting (e.g. triple weighting for NLU tasks), computes an average score per model, and then
    saves the aggregated benchmark results to a JSON file.
    """
    # Dictionary to store scores for each model
    model_scores = defaultdict(list)

    # --- Machine Translation Tasks ---
    mt_eval_results = {}
    # Load the MT evaluation results JSON file (German-English translation)
    file_path = 'Results/mt_all_models_results_de_en.json'
    data = load_json(file_path)
    # Extract the final overall score for each model from the MT evaluation results
    for model, languages in data.items():
        for lang_pair, scores in languages.items():
            final_score = scores.get("FinalGesamtscore")
            if final_score is not None:
                mt_eval_results[model] = final_score
                model_scores[model].append(final_score)

    # --- Summarization Tasks ---
    sum_eval_results = {}
    file_path = "Results/summarization_benchmark_all_models_results.json"
    data = load_json(file_path)
    # Extract the Combined_Score for each model from the summarization evaluation
    for model, details in data.items():
        final_scores = details.get("Final_scores", {})
        combined_score = final_scores.get("Combined_Score")
        if combined_score is not None:
            sum_eval_results[model] = combined_score
            model_scores[model].append(combined_score)

    # --- Dialogue Tasks ---
    dialog_eval_results = {}
    file_path = "Results/llm_dialog_all_models_results.json"
    data = load_json(file_path)
    # Extract the overall dialogue score from the averaged results section
    averaged_results = data.get("averaged_results", {})
    for model, details in averaged_results.items():
        overall = details.get("overall_average", {})
        overall_score = overall.get("overall_score")
        if overall_score is not None:
            dialog_eval_results[model] = overall_score
            model_scores[model].append(overall_score)

    # --- Writing Tasks ---
    # (Commented out; similar approach as above could be applied.)
    # writing_eval_results = {}
    # file_path = "Results/llm_writing_all_models_results.json"
    # data = load_json(file_path)
    # averaged_results = data.get("averaged_results", {})
    # for model, details in averaged_results.items():
    #     overall = details.get("overall_average", {})
    #     overall_score = overall.get("overall_score")
    #     if overall_score is not None:
    #         writing_eval_results[model] = overall_score
    #         model_scores[model].append(overall_score)

    # --- Story Generation Tasks ---
    # (Commented out; similar extraction would be performed.)
    # story_eval_results = {}
    # file_path = "Results/llmstory_all_models_results.json"
    # data = load_json(file_path)
    # averaged_results = data.get("averaged_results", {})
    # for model, details in averaged_results.items():
    #     overall = details.get("overall_average", {})
    #     overall_score = overall.get("overall_score")
    #     if overall_score is not None:
    #         story_eval_results[model] = overall_score
    #         model_scores[model].append(overall_score)

    # --- Combined Writing and Story Generation Tasks ---
    writing_story_results = {}
    story_eval_results = {}
    writing_eval_results = {}
    writing_file_path = "Results/llm_writing_all_models_results.json"
    story_file_path = "Results/llmstory_all_models_results.json"
    writing_data = load_json(writing_file_path)
    story_data = load_json(story_file_path)
    for model, writing_details in writing_data.get("averaged_results", {}).items():
        writing_score = writing_details.get("overall_average", {}).get("overall_score", 0)
        story_score = story_data.get("averaged_results", {}).get(model, {}).get("overall_average", {}).get(
            "overall_score", 0)
        combined_score = (writing_score + story_score) / 2
        story_eval_results[model] = story_score
        writing_eval_results[model] = writing_score
        writing_story_results[model] = combined_score
        model_scores[model].append(combined_score)

    # --- Question Answering Tasks ---
    qa_eval_results = {}
    file_path = "Results/squad_all_models_results.json"
    data = load_json(file_path)
    # For QA, compute the average of exact match and F1 score for each model
    for model, metrics in data.items():
        exact_match = metrics.get("exact_match", 0)
        f1 = metrics.get("f1", 0)
        overall_score = (exact_match + f1) / 2
        qa_eval_results[model] = overall_score
        model_scores[model].append(overall_score)

    # --- Question Generation Tasks ---
    qg_eval_results = {}
    file_path = "Results/squad_qg__all_models_results.json"
    data = load_json(file_path)
    # Extract the Overall_Score from the QG evaluation for each model
    for model, details in data.items():
        final_scores = details.get("Final_scores", {})
        overall_score = final_scores.get("Overall_Score")
        if overall_score is not None:
            qg_eval_results[model] = overall_score
            model_scores[model].append(overall_score)

    # --- NLU Task (SuperGLUE) ---
    superGLUE_results = {}
    file_path = "Results/superGLUE_all_models_results.json"
    data = load_json(file_path)
    # Extract the overall SuperGLUE score for each model and weight it triple
    for model, details in data.items():
        overall_score = details.get("overall_score")
        if overall_score is not None:
            superGLUE_results[model] = overall_score
            model_scores[model].extend([overall_score] * 3)

    # Compute the averaged score for each model
    averaged_scores = {model: sum(scores) / len(scores) for model, scores in model_scores.items() if scores}

    # Aggregate all results into one dictionary
    all_results = {
        "Machine_Translation": mt_eval_results,
        "Summarization": sum_eval_results,
        "Dialog": dialog_eval_results,
        "Writing": writing_eval_results,
        "Storygeneration": story_eval_results,
        "Question_Answering": qa_eval_results,
        "Question_Generation": qg_eval_results,
        "NLU": superGLUE_results,
        "Averaged_Scores": averaged_scores
    }

    output_file = "Results/NLPBenchmarkResults.json"
    save_json(all_results, output_file)

    print("Results saved to:", output_file)
    print("Averaged Model Scores:")
    for model, avg_score in averaged_scores.items():
        print(f"Model: {model}, Averaged Score: {avg_score:.2f}")


def nlp_benchmark_complete(models):
    # TODO: Implement complete benchmark calculation if needed.
    pass
