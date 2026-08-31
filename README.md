# LLM Evaluation Across Legal, Medical, and News-Framing Tasks

This repository contains the code, data-processing scripts, and experimental resources used for a study evaluating the performance of small language models across several challenging natural-language reasoning domains.

The study evaluates locally hosted large language models (LLMs) on **legal reasoning, medical reasoning, and news framing classification** tasks. All models are run locally through **LM Studio**, allowing the experiments to be conducted without relying on proprietary model APIs.

## Overview

The goal of this project is to examine how smaller language models perform across different domains where accurate classification and reasoning are important.

The study evaluates three benchmark datasets:

* **LEGALBENCH** — legal reasoning and classification
* **MedHELM** — medical reasoning and question answering
* **MM-Framing** — news framing and political-leaning classification

For each dataset, **20 questions are randomly selected** and the same questions are evaluated across every model. Model responses are collected automatically, parsed, and compared against the expected answers.

Because the tasks used in this study are evaluated as classification problems, **accuracy** is used as the primary evaluation metric.

## Models

The following locally hosted models are evaluated:

1. **DeepSeek-R1-Distill-Llama-8B**
2. **Gemma 4 E4B**
3. **Qwen3-8B**
4. **Mistral Nemo Instruct 2407**

These models were selected primarily because they are relatively small, instruction-following models that can be accessed locally. Using local models also avoids the cost associated with repeatedly querying commercial APIs.

## Datasets

### LEGALBENCH

LEGALBENCH is used to evaluate legal reasoning capabilities. The benchmark contains a broad collection of legal tasks developed with contributions from legal experts.

For this study, binary classification tasks are used to simplify evaluation and provide a consistent classification format across models.

### MedHELM

MedHELM is used to evaluate model performance on medical reasoning and information-based tasks. The dataset provides a way to examine whether relatively small language models can correctly answer questions in a complex medical domain.

### MM-Framing

The MM-Framing dataset is used to evaluate how models classify the framing and political leaning of news articles.

The study uses a subset of the available questions and evaluates whether models can correctly identify the expected framing classifications.

## Methodology

The experiment follows the same general procedure for each dataset:

1. A dataset is loaded by the corresponding evaluation program.
2. **20 questions are randomly selected.**
3. The selected questions and their expected answers are stored for evaluation.
4. The same questions are presented to each model.
5. Models are accessed locally through LM Studio.
6. Model responses are collected automatically.
7. Responses are parsed to identify the model's predicted answer.
8. Predictions are compared with the expected answers.
9. Accuracy is calculated for each model.

Using the same questions for every model ensures that differences in performance are based on model responses rather than differences in the questions being evaluated.

## Local Model Evaluation

The experiments use **LM Studio's local server** to provide access to the models.

The evaluation programs send prompts to the locally running model and retrieve its response automatically.

The default LM Studio configuration used by the scripts is:

```text
Host: localhost
Port: 1234
```

LM Studio must be running with the appropriate model loaded before an evaluation script is executed.

## Running the Experiments

### Requirements

The experiments require:

* Python 3.x
* LM Studio
* The models being evaluated
* The appropriate benchmark dataset files
* Internet access for initially obtaining datasets/models, if they are not already available locally

Python dependencies used by the evaluation scripts can be installed with:

```bash
pip install -r requirements.txt
```

### Starting LM Studio

1. Open LM Studio.
2. Download or import the model you want to evaluate.
3. Load the model.
4. Start the local server.
5. Confirm that the server is running on the expected host and port.

The evaluation scripts can then be run from the repository.

For example:

```bash
python legalbench_grader.py
```

or the corresponding script for the desired dataset.

## Reproducibility

Several practical considerations should be taken into account when reproducing these experiments.

### Token Limits

Small language models may require more output tokens than the default configuration provides. If the token limit is too low, a model may produce an incomplete answer or fail to provide its prediction in the expected format.

### Timeout Limits

Some models can take a significant amount of time to generate a response, particularly when running locally on limited hardware. Timeout settings may therefore need to be increased when reproducing the experiments.

### Hardware

Model inference speed depends heavily on the hardware available to LM Studio. Differences in GPU, CPU, RAM, and available VRAM can affect how long individual evaluations take.

### Model Configuration

Results may also be affected by model settings such as temperature, context length, and other inference parameters. Reproductions should use the same settings where possible.

### Random Sampling

The evaluation uses randomly selected questions from each dataset. Because the questions are sampled before evaluation and the same selected questions are used for every model, reproductions should preserve the selected question sets when attempting to reproduce the reported results exactly.

## Evaluation Metric

### Accuracy

Accuracy is calculated as:

```text
Accuracy = Correct Predictions / Total Predictions
```

Accuracy was selected as the primary metric because the tasks evaluated in this study use classification-style answers, making the proportion of correctly classified questions a straightforward measure of model performance.

## Purpose of the Repository

This repository is intended to provide the experimental code and supporting resources necessary to understand and reproduce the evaluation performed in the associated research paper.

The repository focuses on:

* Automated LLM evaluation
* Local model inference
* Cross-domain model comparison
* Legal reasoning
* Medical reasoning
* News framing classification
* Reproducible benchmark evaluation

## Citation

If you use this repository or the associated research, please cite the accompanying paper.

```text
Citation information will be added upon publication.
```

## License

License information will be added to this repository.
