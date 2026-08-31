import json
import random
import csv


# -----------------------
# Settings
# -----------------------

INPUT_FILE = "train.json"

NUM_QUESTIONS = 20
RANDOM_SEED = 42

QUESTION_OUTPUT = "selected_questions.jsonl"
ANSWER_OUTPUT = "selected_answers.jsonl"


# -----------------------
# Load JSONL dataset
# -----------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


# -----------------------
# Random selection
# -----------------------

random.seed(RANDOM_SEED)

selected = random.sample(data, NUM_QUESTIONS)


# -----------------------
# Create question JSONL
# Remove answer information
# -----------------------

REMOVE_FIELDS = {
    "cop",
    "exp"
}

questions_only = []

for item in selected:

    question = {
        key: value
        for key, value in item.items()
        if key not in REMOVE_FIELDS
    }

    questions_only.append(question)


with open(QUESTION_OUTPUT, "w", encoding="utf-8") as f:

    for question in questions_only:
        f.write(
            json.dumps(
                question,
                ensure_ascii=False
            )
            + "\n"
        )


# -----------------------
# Create answer JSONL
# -----------------------

option_fields = [
    "opa",
    "opb",
    "opc",
    "opd"
]

option_letters = [
    "A",
    "B",
    "C",
    "D"
]


answers = []

for number, item in enumerate(selected, start=1):

    correct_index = item["cop"] - 1

    answers.append({
        "question_number": number,
        "id": item["id"],
        "correct_option": option_letters[correct_index],
        "correct_index": correct_index,
        "correct_answer": item[option_fields[correct_index]],
        "explanation": item.get("exp", "")
    })


with open(ANSWER_OUTPUT, "w", encoding="utf-8") as f:

    for answer in answers:
        f.write(
            json.dumps(
                answer,
                ensure_ascii=False
            )
            + "\n"
        )


print("Finished!")
print(f"Questions saved to: {QUESTION_OUTPUT}")
print(f"Answers saved to: {ANSWER_OUTPUT}")
