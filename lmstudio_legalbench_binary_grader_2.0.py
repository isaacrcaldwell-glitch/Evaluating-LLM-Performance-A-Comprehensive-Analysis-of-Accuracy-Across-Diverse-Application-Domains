"""
LM Studio LEGALBENCH Binary-Classification Auto-Grader
=======================================================

Sends LEGALBENCH binary-classification questions to a model running in
LM Studio's local OpenAI-compatible server and grades the responses against
an answer key.

IMPORTANT:
- The valid classification choices are NOT hard-coded as "Yes" and "No".
- They are read from the `correct_answer` values in the answers JSONL file.
- Choices are determined separately for each task when possible.
- The model is explicitly shown the valid choices for that task.
- The grader records both the raw model response and the extracted label.

Expected questions file (JSONL):
    {"id": "...", "task": "...", "question": "..."}

Expected answers file (JSONL):
    {"id": "...", "task": "...", "correct_answer": "..."}

Run:
    python lmstudio_legalbench_binary_grader.py

Or:
    python lmstudio_legalbench_binary_grader.py \
        --questions questions.jsonl \
        --answers answers.jsonl \
        --output graded_results.json

Optional:
    --model "model-identifier"
    --temperature 0
    --max-tokens 30
    --timeout 120
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

try:
    import requests
except ImportError:
    print("Missing dependency. Install it with:")
    print("    pip install requests")
    sys.exit(1)


DEFAULT_HOST = "http://localhost:1234"
CHAT_ENDPOINT = "/v1/chat/completions"
MODELS_ENDPOINT = "/v1/models"


def load_jsonl(path):
    """Load one JSON object per line."""
    if not os.path.exists(path):
        print(f"ERROR: {path} was not found.")
        sys.exit(1)

    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"WARNING: Skipping malformed line {line_num} "
                    f"in {path}: {e}"
                )

    return items


def get_loaded_model(host, timeout=10):
    """Get the model currently loaded in LM Studio."""
    try:
        response = requests.get(
            host + MODELS_ENDPOINT,
            timeout=timeout
        )
        response.raise_for_status()

        models = [
            model["id"]
            for model in response.json().get("data", [])
        ]

        if not models:
            print(
                "LM Studio reports no loaded models. "
                "Load a model first."
            )
            sys.exit(1)

        if len(models) > 1:
            print(
                f"Multiple models available: {models}. "
                f"Using '{models[0]}'."
            )

        return models[0]

    except requests.RequestException as e:
        print(
            f"Could not reach LM Studio at {host}.\n"
            f"Make sure the Developer server is running.\n"
            f"Error: {e}"
        )
        sys.exit(1)


def normalize_label(label):
    """
    Normalize a classification label for comparison.

    This is intentionally conservative so labels such as:
        entailment
        contradiction
        positive
        negative
        Yes
        No
        0
        1

    can all be handled without changing their meaning.
    """
    return str(label).strip().casefold()


def build_task_choices(answer_rows):
    """
    Build:
        task -> list of valid classification labels

    The choices come directly from correct_answer values in the
    answer document.
    """
    task_choices = defaultdict(list)

    for row in answer_rows:
        task = str(row.get("task", "")).strip()
        answer = row.get("correct_answer")

        if answer is None:
            continue

        answer = str(answer).strip()

        if not answer:
            continue

        # Avoid duplicate labels while preserving original spelling/order.
        normalized = normalize_label(answer)

        existing = {
            normalize_label(x)
            for x in task_choices[task]
        }

        if normalized not in existing:
            task_choices[task].append(answer)

    return dict(task_choices)


def build_global_choices(answer_rows):
    """Build a fallback set of all labels found in the answer document."""
    choices = []

    for row in answer_rows:
        answer = row.get("correct_answer")

        if answer is None:
            continue

        answer = str(answer).strip()

        if not answer:
            continue

        normalized = normalize_label(answer)

        if normalized not in {
            normalize_label(x) for x in choices
        }:
            choices.append(answer)

    return choices


def build_prompt(question_row, choices):
    """
    Build a prompt that explicitly tells the model which classification
    labels it is allowed to return.

    We deliberately do NOT rely on the question's own "Yes or No"
    instruction because LEGALBENCH contains tasks whose binary labels
    may use other terminology.
    """
    question = str(question_row.get("question", "")).strip()

    choice_text = "\n".join(
        f"- {choice}"
        for choice in choices
    )

    return (
        "Classify the following LEGALBENCH example.\n\n"
        "You MUST choose exactly one of the following valid labels:\n"
        f"{choice_text}\n\n"
        "Return ONLY the label itself. Do not explain your answer.\n\n"
        "Question:\n"
        f"{question}"
    )


def build_system_instruction(choices):
    choice_text = ", ".join(f'"{choice}"' for choice in choices)

    return (
        "You are answering a LEGALBENCH binary-classification task. "
        f"The only valid answer labels are: {choice_text}. "
        "Return exactly one of those labels and nothing else. "
        "Do not provide reasoning or explanation."
    )


def ask_model(
    host,
    model,
    prompt,
    system_instruction,
    temperature,
    max_tokens,
    timeout
):
    """Send one question to LM Studio and print the raw API response."""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_instruction
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        host + CHAT_ENDPOINT,
        json=payload,
        timeout=timeout
    )

    response.raise_for_status()

    # ------------------------------------------------------------
    # PRINT THE COMPLETE RAW RESPONSE FROM LM STUDIO
    # This is printed for EVERY question, regardless of whether the
    # response can be parsed into a valid classification label.
    # ------------------------------------------------------------

    data = response.json()

    print()
    print("    ========================================")
    print("    RAW LM STUDIO JSON RESPONSE (EVERY QUESTION)")
    print("    ========================================")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("    ========================================")
    print()

    # ------------------------------------------------------------
    # EXTRACT THE MODEL'S ACTUAL TEXT
    # ------------------------------------------------------------

    try:
        content = data["choices"][0]["message"]["content"]

    except (KeyError, IndexError, TypeError) as e:

        print()
        print("    !!! COULD NOT FIND MODEL CONTENT !!!")
        print(f"    Error: {e}")
        print("    The complete LM Studio response is shown above.")
        print()

        return ""

    # ------------------------------------------------------------
    # HANDLE EMPTY / NONE CONTENT
    # ------------------------------------------------------------

    if content is None:

        print()
        print("    !!! LM STUDIO RETURNED NO MESSAGE CONTENT !!!")
        print()

        return ""

    # ------------------------------------------------------------
    # HANDLE NON-STRING CONTENT
    # ------------------------------------------------------------

    if not isinstance(content, str):

        if isinstance(content, list):

            pieces = []

            for item in content:

                if isinstance(item, dict) and "text" in item:
                    pieces.append(str(item["text"]))

                else:
                    pieces.append(str(item))

            content = "".join(pieces)

        else:
            content = str(content)

    content = content.strip()

    # ------------------------------------------------------------
    # PRINT THE ACTUAL TEXT THAT THE PARSER RECEIVES
    # ------------------------------------------------------------

    print()
    print("    ========================================")
    print("    MODEL TEXT USED BY PARSER")
    print("    ========================================")
    print(repr(content))
    print("    ========================================")
    print()

    return content

def extract_label(raw_answer, choices):
    """
    Extract one of the allowed labels from the model's response.

    Priority:
    1. Exact match
    2. Quoted/marked exact label
    3. Standalone occurrence of an allowed label
    4. None if no valid label can be identified

    The longest labels are checked first so labels such as
    "not applicable" are not accidentally interpreted as "applicable".
    """
    text = str(raw_answer).strip()

    if not text:
        return None

    # 1. Exact match.
    for choice in choices:
        if normalize_label(text) == normalize_label(choice):
            return choice

    # 2. Remove common surrounding formatting.
    cleaned = text.strip(" \t\r\n\"'`*_[]()")

    for choice in choices:
        if normalize_label(cleaned) == normalize_label(choice):
            return choice

    # 3. Search for the allowed label in the response.
    #
    # Longer labels first prevents partial matches.
    sorted_choices = sorted(
        choices,
        key=lambda x: len(str(x)),
        reverse=True
    )

    for choice in sorted_choices:
        choice_text = str(choice).strip()

        # For labels consisting of normal words/numbers, use boundaries.
        if re.fullmatch(r"[\w\s.+\-/]+", choice_text, re.UNICODE):
            pattern = (
                r"(?<!\w)"
                + re.escape(choice_text)
                + r"(?!\w)"
            )
        else:
            pattern = re.escape(choice_text)

        if re.search(pattern, text, re.IGNORECASE):
            return choice

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Grade LEGALBENCH binary-classification tasks using LM Studio."
    )

    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="LM Studio server URL."
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier. If omitted, uses the loaded model."
    )

    parser.add_argument(
        "--questions",
        default="questions.jsonl",
        help="LEGALBENCH questions JSONL file."
    )

    parser.add_argument(
        "--answers",
        default="answers.jsonl",
        help="LEGALBENCH answer-key JSONL file."
    )

    parser.add_argument(
        "--output",
        default="graded_results.json",
        help="Output JSON file."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature."
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Maximum generated tokens per answer."
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Seconds to wait for each model response."
    )

    args = parser.parse_args()

    questions = load_jsonl(args.questions)
    answer_rows = load_jsonl(args.answers)

    # Match answers by ID, not by position.
    answer_key = {
        row["id"]: row
        for row in answer_rows
        if "id" in row
    }

    task_choices = build_task_choices(answer_rows)
    global_choices = build_global_choices(answer_rows)

    print(f"Loaded {len(questions)} questions.")
    print(f"Loaded {len(answer_key)} answer-key entries.")
    print()

    print("Classification labels found in answers document:")

    for task, choices in sorted(task_choices.items()):
        print(f"  {task}: {choices}")

    print()

    # Warn if the answer document does not actually contain two labels
    # for a supposedly binary task.
    for task, choices in sorted(task_choices.items()):
        if len(choices) != 2:
            print(
                f"WARNING: Task '{task}' has {len(choices)} "
                f"unique answer label(s): {choices}"
            )

    print()

    model = args.model or get_loaded_model(args.host)

    print(f"Using model: {model}")
    print()

    results = []

    correct_count = 0
    graded_count = 0
    unrecognized_count = 0

    per_task = defaultdict(lambda: [0, 0])

    for index, question_row in enumerate(questions, 1):
        qid = question_row.get("id")
        task = str(
            question_row.get("task", "Unknown")
        ).strip() or "Unknown"

        answer_row = answer_key.get(qid)

        # Prefer labels associated with this task.
        choices = task_choices.get(task, [])

        # If the task could not be identified, use all labels.
        if not choices:
            choices = global_choices

        print(
            f"[{index}/{len(questions)}] "
            f"task={task} id={qid}"
        )

        if not choices:
            print("    ERROR: No classification labels found.")
            results.append({
                "id": qid,
                "task": task,
                "question": question_row.get("question"),
                "model_raw_answer": None,
                "model_answer": None,
                "correct_answer": (
                    answer_row.get("correct_answer")
                    if answer_row else None
                ),
                "is_correct": None,
                "error": "no_classification_labels_found"
            })
            continue

        prompt = build_prompt(question_row, choices)
        system_instruction = build_system_instruction(choices)

        try:
            raw_answer = ask_model(
                args.host,
                model,
                prompt,
                system_instruction,
                args.temperature,
                args.max_tokens,
                args.timeout
            )

        except requests.RequestException as e:
            print(f"    ERROR querying model: {e}")

            results.append({
                "id": qid,
                "task": task,
                "question": question_row.get("question"),
                "model_raw_answer": None,
                "model_answer": None,
                "correct_answer": (
                    answer_row.get("correct_answer")
                    if answer_row else None
                ),
                "is_correct": False,
                "error": "request_failed",
                "error_message": str(e)
            })

            continue

        model_answer = extract_label(raw_answer, choices)

        correct_answer = None

        if answer_row is not None:
            correct_answer = str(
                answer_row.get("correct_answer", "")
            ).strip()

        if answer_row is None:
            is_correct = None
            print("    WARNING: No answer-key entry for this ID.")

        elif model_answer is None:
            is_correct = False
            unrecognized_count += 1
            print(
                f"    model said: {raw_answer!r} "
                f"-> UNRECOGNIZED | correct: {correct_answer!r}"
            )

        else:
            is_correct = (
                normalize_label(model_answer)
                == normalize_label(correct_answer)
            )

            print(
                f"    model said: {raw_answer!r} "
                f"-> {model_answer!r} | "
                f"correct: {correct_answer!r} | "
                f"{'CORRECT' if is_correct else 'WRONG'}"
            )

        results.append({
            "id": qid,
            "task": task,
            "question": question_row.get("question"),
            "valid_choices": choices,
            "model_raw_answer": raw_answer,
            "model_answer": model_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        })

        if is_correct is not None:
            graded_count += 1
            correct_count += int(is_correct)

            per_task[task][1] += 1
            per_task[task][0] += int(is_correct)

        # Save after every question so an interrupted run retains progress.
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=== Results ===")

    if graded_count:
        accuracy = 100 * correct_count / graded_count
        print(
            f"Overall accuracy: "
            f"{correct_count}/{graded_count} "
            f"({accuracy:.1f}%)"
        )

    print(f"Unrecognized model answers: {unrecognized_count}")

    print()
    print("Per-task breakdown:")

    for task, (correct, total) in sorted(per_task.items()):
        if total:
            accuracy = 100 * correct / total
            print(
                f"  {task}: "
                f"{correct}/{total} "
                f"({accuracy:.1f}%)"
            )

    print()
    print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
