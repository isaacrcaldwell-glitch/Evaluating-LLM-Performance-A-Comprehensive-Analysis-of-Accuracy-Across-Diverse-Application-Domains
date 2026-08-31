"""
LM Studio Multiple-Choice Auto-Grader
========================================
Sends 4-option multiple-choice questions to a locally-downloaded model
running in LM Studio's local server, and auto-grades the responses
against an answer key — matched by question `id` (UUID), not position,
so the two files don't need to stay in the same order.

------------------------------------------------------------------
SETUP
------------------------------------------------------------------
1. In LM Studio: load your model, go to the "Developer" tab, click
   "Start Server" (default: http://localhost:1234). Leave LM Studio
   running — this script talks to it over localhost.
2. pip install requests

------------------------------------------------------------------
EXPECTED INPUT FORMAT (JSON Lines — one JSON object per line)
------------------------------------------------------------------
questions file, e.g.:
    {"question": "...", "opa": "...", "opb": "...", "opc": "...",
     "opd": "...", "subject_name": "Pathology", "topic_name": null,
     "id": "c1bb1cd3-...", "choice_type": "single"}

answers file, e.g.:
    {"question_number": 1, "id": "c1bb1cd3-...", "correct_option": "D",
     "correct_index": 3, "correct_answer": "All the above",
     "explanation": "..."}

Matching is done on the shared "id" field. correct_option (A/B/C/D)
is what's used for grading; the explanation field is loaded but never
shown to the model.

------------------------------------------------------------------
RUNNING
------------------------------------------------------------------
    python lmstudio_mcq_grader.py
    python lmstudio_mcq_grader.py --questions selected_questions.jsonl --answers selected_answers.jsonl
    python lmstudio_mcq_grader.py --model "your-model-identifier"

Writes graded_results.json (full detail) as it goes, and prints
overall + per-subject accuracy at the end.
"""

import argparse
import json
import os
import re
import sys

try:
    import requests
except ImportError:
    print("Missing dependency. Run: pip install requests")
    sys.exit(1)


DEFAULT_HOST = "http://localhost:1234"
CHAT_ENDPOINT = "/v1/chat/completions"
MODELS_ENDPOINT = "/v1/models"

OPTION_KEYS = ["opa", "opb", "opc", "opd"]
OPTION_LETTERS = ["A", "B", "C", "D"]

SYSTEM_INSTRUCTION = (
    "You are answering multiple-choice medical exam questions. Respond "
    "with ONLY the single letter of the correct option (A, B, C, or D). "
    "Do not include explanations, reasoning, or any other text."
)

# Strip reasoning-model "thinking" blocks before we try to extract a letter.
# Covers the common tag styles: <think>...</think>, <thinking>...</thinking>,
# and the unclosed case where a response gets cut off mid-think (no closing
# tag at all, because max_tokens ran out).
THINK_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_UNCLOSED_RE = re.compile(r"<think(?:ing)?>.*$", re.DOTALL | re.IGNORECASE)


def load_jsonl(path):
    if not os.path.exists(path):
        print(f"{path} not found.")
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
                print(f"  Skipping malformed line {line_num} in {path}: {e}")
    return items


def get_loaded_model(host, timeout=10):
    try:
        resp = requests.get(host + MODELS_ENDPOINT, timeout=timeout)
        resp.raise_for_status()
        models = [m["id"] for m in resp.json().get("data", [])]
        if not models:
            print("LM Studio reports no loaded models. Load a model in LM Studio first.")
            sys.exit(1)
        if len(models) > 1:
            print(f"Multiple models available: {models}. Using '{models[0]}' "
                  f"(pass --model to choose a specific one).")
        return models[0]
    except requests.RequestException as e:
        print(f"Could not reach LM Studio server at {host}. Is the local "
              f"server started (Developer tab -> Start Server)? Error: {e}")
        sys.exit(1)


def build_prompt(q, no_think=False):
    lines = [q["question"].strip(), ""]
    for letter, key in zip(OPTION_LETTERS, OPTION_KEYS):
        option_text = str(q.get(key, "")).strip()
        lines.append(f"{letter}. {option_text}")
    lines.append("")
    lines.append("Answer with only the letter of the correct option (A, B, C, or D).")
    if no_think:
        # Qwen3's chat template honors this directive to skip its extended
        # reasoning phase and answer directly — avoids burning the whole
        # max_tokens budget on <think>/reasoning_content before ever
        # reaching a letter. Harmless no-op for models that don't support it.
        lines.append("/no_think")
    return "\n".join(lines)


def clean_response_text(text):
    """Remove reasoning/thinking blocks a 'thinking' model may prepend to
    its answer, including the case where the block never got closed
    because max_tokens ran out mid-thought."""
    if not text:
        return text
    cleaned = THINK_BLOCK_RE.sub("", text)
    if cleaned == text and re.search(r"<think(?:ing)?>", text, re.IGNORECASE):
        # Opening tag present but no closing tag -> truncated mid-think.
        cleaned = THINK_OPEN_UNCLOSED_RE.sub("", text)
    return cleaned.strip()


def ask_model(host, model, prompt, temperature, max_tokens, timeout, debug=False):
    resp = requests.post(host + CHAT_ENDPOINT, json={
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    message = choice.get("message", {}) or {}
    content = (message.get("content") or "").strip()
    # Some reasoning-model servers (Qwen3 and others) put the "thinking"
    # text in a separate reasoning_content field rather than <think> tags
    # inside content. If content came back empty — e.g. the completion got
    # cut off (finish_reason == "length") before the model finished
    # thinking — fall back to it so extract_letter has something to search,
    # on the off chance a letter is mentioned in the reasoning itself.
    reasoning_content = (message.get("reasoning_content") or "").strip()
    finish_reason = choice.get("finish_reason")
    tool_calls = message.get("tool_calls")

    # Always print the raw model response so it can be inspected even when
    # the response parses successfully. This is intentionally unconditional.
    print(f"    [raw] finish_reason={finish_reason!r} "
          f"tool_calls={tool_calls!r} "
          f"reasoning_tokens={data.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens')!r}")
    print(f"    [raw] content: {content!r}")
    if reasoning_content:
        print(f"    [raw] reasoning_content: {reasoning_content!r}")
    print(f"    [raw] full response JSON: {json.dumps(data)}")

    effective_content = content or reasoning_content
    return effective_content, finish_reason, tool_calls, data


def dump_raw_response(data):
    """Print the full API response verbatim. Called when we couldn't pull a
    letter out of the answer, so we can see exactly what the model/server
    sent back — e.g. an empty '()' shell from a broken tool-call/chat
    template, a separate reasoning_content field, non-stop finish_reason,
    etc. — instead of just the empty-looking content string."""
    print(f"    [no-letter-debug] full raw response JSON: {json.dumps(data)}")


def extract_letter(raw_answer, q):
    """Pull a single A/B/C/D choice out of the model's raw response,
    tolerating short reasoning, punctuation, or 'Option A' phrasing."""
    text = raw_answer.strip()

    # 1) A lone letter, optionally with punctuation: "A", "A.", "(A)"
    m = re.match(r"^\(?([ABCD])\)?[\.\):,]?\s*$", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 2) First standalone A/B/C/D token anywhere in the response
    m = re.search(r"\b([ABCD])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3) Fall back to matching the full option text verbatim
    text_lower = text.lower()
    for letter, key in zip(OPTION_LETTERS, OPTION_KEYS):
        option_text = str(q.get(key, "")).strip().lower()
        if option_text and option_text in text_lower:
            return letter

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default=None,
                         help="Model identifier as shown in LM Studio. "
                              "If omitted, uses whichever model is currently loaded.")
    parser.add_argument("--questions", default="selected_questions.jsonl")
    parser.add_argument("--answers", default="selected_answers.jsonl")
    parser.add_argument("--output", default="graded_results.json")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=120,
                         help="Seconds to wait per question (local models can be slow)")
    parser.add_argument("--debug", action="store_true",
                         help="Print the full raw API response for every question, "
                              "not just ones that look empty/broken.")
    parser.add_argument("--no-think", action="store_true",
                         help="Append '/no_think' to the prompt to skip Qwen3's "
                              "extended reasoning phase and get a direct answer. "
                              "Harmless no-op on models that don't support it.")
    parser.add_argument("--append", action="store_true",
                         help="Merge into --output instead of overwriting it. Existing "
                              "rows are matched on (question id, model) — rerunning the "
                              "same model over the same question replaces that row; "
                              "results from other models/questions are kept untouched.")
    args = parser.parse_args()

    questions = load_jsonl(args.questions)
    answer_rows = load_jsonl(args.answers)
    answer_key = {row["id"]: row for row in answer_rows if "id" in row}

    model = args.model or get_loaded_model(args.host)
    print(f"Using model: {model}")
    print(f"Loaded {len(questions)} questions, {len(answer_key)} answer key entries.\n")

    # results_by_key holds EVERY row that should end up in the output file —
    # including, in --append mode, rows loaded from a prior run against
    # other models. Keyed on (id, model) so different models' answers to
    # the same question never clobber each other.
    results_by_key = {}
    if args.append and os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for row in existing:
                key = (row.get("id"), row.get("model", "unknown"))
                results_by_key[key] = row
            print(f"--append: loaded {len(existing)} existing rows from {args.output}\n")
        except (json.JSONDecodeError, OSError) as e:
            print(f"--append: couldn't read existing {args.output} ({e}), starting fresh.\n")

    for i, q in enumerate(questions, 1):
        qid = q.get("id")
        subject = q.get("subject_name") or "Unknown"
        answer_row = answer_key.get(qid)

        prompt = build_prompt(q, no_think=args.no_think)
        print(f"[{i}/{len(questions)}] ({subject}) id={qid} asking...")

        try:
            raw_answer, finish_reason, tool_calls, raw_data = ask_model(
                args.host, model, prompt, args.temperature, args.max_tokens,
                args.timeout, debug=args.debug)
        except requests.RequestException as e:
            print(f"    Error querying model: {e}")
            results_by_key[(qid, model)] = {
                "id": qid, "model": model, "subject": subject, "question": q.get("question"),
                "model_raw_answer": None, "model_letter": None,
                "correct_option": answer_row.get("correct_option") if answer_row else None,
                "is_correct": False, "error": "request_failed",
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(list(results_by_key.values()), f, indent=2)
            continue

        cleaned_answer = clean_response_text(raw_answer)
        model_letter = extract_letter(cleaned_answer, q)

        if model_letter is None and not args.debug:
            # We already dumped raw JSON above if content was empty/had
            # tool_calls; this covers the remaining case you hit — content
            # was present (e.g. literal "()") but had no usable letter in it.
            dump_raw_response(raw_data)

        if answer_row is None:
            print(f"    (no answer key entry for id={qid}, skipping grading)")
            results_by_key[(qid, model)] = {
                "id": qid, "model": model, "subject": subject, "question": q.get("question"),
                "model_raw_answer": raw_answer, "model_letter": model_letter,
                "correct_option": None, "is_correct": None,
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(list(results_by_key.values()), f, indent=2)
            continue

        correct_option = str(answer_row.get("correct_option", "")).strip().upper()
        is_correct = model_letter is not None and model_letter == correct_option

        print(f"    model said: {raw_answer!r} -> {model_letter}  |  "
              f"correct: {correct_option}  |  {'CORRECT' if is_correct else 'WRONG'}"
              f"{'  |  finish_reason=' + repr(finish_reason) if finish_reason not in (None, 'stop') else ''}")

        results_by_key[(qid, model)] = {
            "id": qid, "model": model, "subject": subject, "question": q.get("question"),
            "model_raw_answer": raw_answer, "model_letter": model_letter,
            "finish_reason": finish_reason,
            "correct_option": correct_option, "correct_answer_text": answer_row.get("correct_answer"),
            "is_correct": is_correct,
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(list(results_by_key.values()), f, indent=2)

    print("\n=== Results ===")
    this_run_rows = [row for row in results_by_key.values() if row.get("model") == model]
    graded_rows = [row for row in this_run_rows if row.get("is_correct") is not None]
    correct_count = sum(1 for row in graded_rows if row["is_correct"])
    graded_count = len(graded_rows)
    if graded_count:
        print(f"Overall accuracy ({model}): {correct_count}/{graded_count} "
              f"({100*correct_count/graded_count:.1f}%)")

    per_subject = {}  # subject -> [correct, total]
    for row in graded_rows:
        subject = row.get("subject") or "Unknown"
        per_subject.setdefault(subject, [0, 0])
        per_subject[subject][1] += 1
        per_subject[subject][0] += int(row["is_correct"])

    print("\nPer-subject breakdown:")
    for subject, (c, t) in sorted(per_subject.items()):
        print(f"  {subject}: {c}/{t}")

    print(f"\nFull results saved to {args.output}"
          f" ({len(results_by_key)} total rows across all models in this file)")


if __name__ == "__main__":
    main()
