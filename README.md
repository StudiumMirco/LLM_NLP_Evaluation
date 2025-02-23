# LLM_NLP_Evaluation
## A Comparison of the Linguistic Abilities of Large Language Models: Analysis and Evaluation

### Abstract

This thesis analyzes and compares the linguistic capabilities of the most prominent Large Language Models (LLMs) currently available. The focus lies on "General-Purpose" models such as GPT-4o, Claude-3.5, and Gemini, which are versatile and widely used. The study aims to evaluate these models in areas like natural language understanding, machine translation, summarization, and dialog generation. Modern metrics, including LLM-as-a-Judge approaches, are employed to better simulate human judgment. The research also investigates the impact of model architectures and training data on performance. Results reveal significant differences in linguistic processing abilities, particularly in natural language understanding and question generation. Based on the results, strengths and weaknesses are identified for each tested model. This thesis not only provides a snapshot of current performance but also offers a scalable methodology for future comparisons. It thereby lays the foundation for more effective and responsible use of LLMs in practical applications.

### Purpose

This repository hosts the code and the data used for evaluating the linguistic abilities of Large Language Models. Additionally, the repository contains the experiment's results in numerical form and as illustrative graphs in JSON and xlsx files.

### Structure of the Repository

The repository contains the complete code to reproduce the experiment evaluating the language capabilities of large language models.

    * In the CondaEnv folder, you will find the backup of the Conda environment prepared for this code.
    * In the Benchmarks folder, there are scripts that use several metrics and datasets to calculate the results for an entire task area, such as summarization.
    * In the Metrics folder, you will find the scripts for the individual metrics used.
    * In the Models folder, there is the script for integrating the used models.
    * In the dataload folder, you will find the scripts for integrating the used datasets.
    * In the main script, you can select the models as well as the desired benchmarks/metrics. Running the main method starts the script.
    * The Utils script contains helper methods.

Apart from the code, the data folder contains datasets that were not available via the Huggingface API and were integrated manually. In the Predictions_Results_Errors folder, you will find the results of the two experiment runs along with the model predictions, error logs, and processed tables and visualizations.


### Models, Test Areas, and Metrics Used


### Overall Results


### How to use
conda Umgebung einrichten

Api keys einsetzen

Auswahl Modelle und Prüfbereiche in Main

mögliches erweitern:
Datensätze, Metriken, Modelle etc.



