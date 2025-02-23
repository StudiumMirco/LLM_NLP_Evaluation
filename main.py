# Import evaluation functions from benchmark modules
from Benchmarks.mt_benchmark import mt_eval
from Benchmarks.nlp_benchmark import nlp_benchmark
from Benchmarks.summarization_benchmark import summarization_eval
from Benchmarks.superGLUE import superGLUE

# Import evaluation functions for specific LLM tasks
from Metrics.llm_eval_dialog import llm_dialog_eval
from Metrics.llm_mtbench import llm_writing_eval
from Metrics.llm_storygeneration import llm_story_eval
from Metrics.squadv2 import squad_eval
from Metrics.squadv2_question_generation import squad_eval_qg

# Import model classes for different LLM provider
from Models.models import OpenAIModel
from Models.models import GeminiModel
from Models.models import AnthropicModel
from Models.models import LLamaModel


# !!! Set API-Keys in models.py !!!

if __name__ == '__main__':
    # Instantiate OpenAI models with specified parameters:
    gpt4o = OpenAIModel(
        model_name="gpt-4o",
        max_new_tokens=1000,
        temperature=0,
        system_prompt=None
    )  # GPT-4 variant "gpt-4o" with zero temperature and a token limit of 1000

    modelo1 = OpenAIModel(
        model_name="o1-preview",
        max_new_tokens=1000,
        temperature=0,
        system_prompt=None
    )  # OpenAI model "o1-preview" with similar parameters

    # Instantiate Gemini models:
    gemini_flash = GeminiModel(
        model_name="gemini-1.5-flash",
        max_new_tokens=1000,
        temperature=0
    )  # Gemini model variant "gemini-1.5-flash"

    gemini_pro = GeminiModel(
        model_name="gemini-1.5-pro",
        max_new_tokens=1000,
        temperature=0
    )  # Gemini model variant "gemini-1.5-pro"

    # Instantiate Anthropic models:
    haiku = AnthropicModel(
        model_name="claude-3-5-haiku-20241022",
        max_new_tokens=1000,
        temperature=0
    )  # Anthropic model "claude-3-5-haiku-20241022"

    claude_35 = AnthropicModel(
        model_name="claude-3-5-sonnet-20241022",
        max_new_tokens=1000,
        temperature=0
    )  # Anthropic model "claude-3-5-sonnet-20241022"

    # Instantiate LLama models:
    llama32 = LLamaModel(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        max_new_tokens=1000,
        temperature=0
    )  # LLaMA model "Llama-3.2-3B-Instruct"

    llama31 = LLamaModel(
        model_name="meta-llama/Llama-3.1-70B-Instruct",
        max_new_tokens=1000,
        temperature=0
    )  # LLaMA model "Llama-3.1-70B-Instruct"

    # Create a list of all models for iterative evaluation
    models = [
        gpt4o,
        modelo1,
        gemini_flash,
        gemini_pro,
        haiku,
        claude_35,
        llama32,
        llama31
    ]

    # Run various benchmarks on the list of models:
    superGLUE(models)  # Evaluate models on the SuperGLUE benchmark
    mt_eval(models, subset_length=1000)  # Evaluate machine translation performance using 1000 samples
    summarization_eval(models, subset_length=250)  # Evaluate summarization with a subset of 250 samples
    llm_dialog_eval(models, subset_length=-1)  # Evaluate dialog performance using the full dataset (subset_length=-1)
    llm_writing_eval(models)  # Evaluate writing capabilities of the models
    llm_story_eval(models, subset_length=50)  # Evaluate story generation on a subset of 50 samples
    squad_eval(models, subset_length=1000)  # Evaluate models on SQuAD v2 with 1000 samples
    squad_eval_qg(models, subset_length=500)  # Evaluate question generation on SQuAD with 500 samples
    nlp_benchmark()  # Run general NLP benchmarks
