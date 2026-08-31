import json
import ast
import argparse
import re
from pathlib import Path

import requests


# ============================================================
# CONFIGURATION
# ============================================================

LM_STUDIO_HOST = "localhost"
LM_STUDIO_PORT = 1234

# Change this if you want to specify a particular model.
MODEL_NAME = "google/gemma-4-e4b"


def lm_studio_base_url(host=LM_STUDIO_HOST, port=LM_STUDIO_PORT):
    return f"http://{host}:{port}"


def model_tagged_filename(base_path, model_name):
    """
    Turn a base output path like 'mm_framing_results.json' plus a model
    name like 'qwen2.5-7b-instruct' into 'mm_framing_results_qwen2.5-7b-instruct.json',
    so results from different models never overwrite each other and the
    filename tells you at a glance which model produced it.
    """
    base_path = Path(base_path)
    stem = base_path.stem
    suffix = base_path.suffix or ".json"
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    return str(base_path.with_name(f"{stem}_{safe_model}{suffix}"))


def check_lm_studio_connection(base_url):
    """
    Hits LM Studio's /v1/models endpoint before doing any real work, so a
    dead server fails fast with an actionable message instead of a raw
    ConnectionError buried after the script has already started processing.
    """

    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("data", [])

    except requests.exceptions.ConnectionError:
        raise SystemExit(
            f"\nCould not connect to LM Studio at {base_url}.\n\n"
            "Checklist:\n"
            "  1. Open LM Studio -> Developer tab -> make sure the local "
            "server is started (not just a model loaded).\n"
            "  2. Confirm the port shown in LM Studio matches what you "
            f"passed here ({base_url}). Default is 1234.\n"
            "  3. If this script is running somewhere OTHER than the same "
            "machine as LM Studio (e.g. WSL, a Docker container, a remote "
            "server, or a VM), 'localhost' refers to that machine, not "
            "the one running LM Studio. Use --host with LM Studio's "
            "machine IP instead (on WSL2 talking to Windows-side LM "
            "Studio, try --host $(cat /etc/resolv.conf | grep nameserver "
            "| awk '{print $2}'), or enable 'Serve on Local Network' in "
            "LM Studio and use that IP).\n"
            "  4. Check any firewall that might be blocking the port.\n"
        )

    except requests.exceptions.RequestException as e:
        raise SystemExit(f"\nLM Studio responded but something went wrong: {e}\n")

    if not models:
        print(
            f"WARNING: Connected to LM Studio at {base_url}, but no model "
            "is currently loaded. Load a model in LM Studio before "
            "continuing, or requests will fail.\n"
        )
    else:
        loaded = ", ".join(m.get("id", "?") for m in models)
        print(f"Connected to LM Studio at {base_url}. Loaded model(s): {loaded}\n")

    return [m.get("id") for m in models]


# ============================================================
# MM-FRAMING LABELS
# ============================================================
#
# The Hugging Face dataset uses the full names, while your
# framing.jsonl uses abbreviated labels.
#
# The LLM is shown the full names AND the exact label it must
# return. This makes grading straightforward.
#
# These are the 14 generic framing categories.
# ============================================================

FRAMING_LABELS = {
    "economic": "Economic",
    "capacity": "Capacity and resources",
    "morality": "Morality",
    "fairness": "Fairness and equality",
    "legality": "Legality, constitutionality and jurisprudence",
    "policy": "Policy prescription and evaluation",
    "crime": "Crime and punishment",
    "security": "Security and defense",
    "health": "Health and safety",
    "quality_life": "Quality of life",
    "culture": "Cultural identity",
    "public_op": "Public opinion",
    "political": "Political",
    "regulation": "External regulation and reputation",
}

VALID_FRAME_LABELS = set(FRAMING_LABELS.keys())


# ============================================================
# POLITICAL LEANINGS
# ============================================================

POLITICAL_LEANINGS = [
    "left",
    "left_lean",
    "center",
    "right_lean",
    "right",
]


# ============================================================
# FILE LOADING
# ============================================================

def load_articles(filename):
    """
    articles.json is a dictionary where the UUID is the key.

    Example:

    {
        "uuid-here": {
            "title": "...",
            "political_leaning": "left",
            "text": "..."
        }
    }
    """

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "articles.json should contain a JSON object "
            "with UUIDs as keys."
        )

    return data


def load_framing_jsonl(filename):
    """
    Loads framing.jsonl.

    Each line is one JSON object.
    """

    records = []

    with open(filename, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as e:
                print(
                    f"WARNING: Could not parse line "
                    f"{line_number}: {e}"
                )

    return records


# ============================================================
# FRAME PARSING
# ============================================================

def parse_frames(value):
    """
    Converts the representation used in framing.jsonl into
    a normal Python list.

    Example:

        "['morality', 'crime', 'security']"

    becomes:

        ['morality', 'crime', 'security']
    """

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    value = str(value).strip()

    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                str(x).strip()
                for x in parsed
                if str(x).strip()
            ]

    except Exception:
        pass

    # Fallback in case the field isn't formatted as a
    # Python list.
    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def normalize(value):
    """
    Normalizes strings for comparison.
    """

    if value is None:
        return ""

    value = str(value).strip().lower()

    value = re.sub(r"\s+", " ", value)

    return value


# ============================================================
# GROUND TRUTH
# ============================================================

def build_ground_truth(framing_records):
    """
    Creates:

        UUID -> framing labels

    from framing.jsonl.
    """

    ground_truth = {}

    for record in framing_records:

        uuid = record.get("uuid")

        if not uuid:
            continue

        frames = parse_frames(
            record.get("text-generic-frame")
        )

        ground_truth[str(uuid)] = frames

    return ground_truth


# ============================================================
# PROMPT
# ============================================================

def build_prompt(article_text):

    framing_options = "\n".join(
        f'- "{short}" = {long_name}'
        for short, long_name in FRAMING_LABELS.items()
    )

    leaning_options = "\n".join(
        f'- "{leaning}"'
        for leaning in POLITICAL_LEANINGS
    )

    return f"""
You are classifying a news article according to the
MM-Framing dataset.

You must perform TWO independent classifications.

============================================================
TASK 1: GENERIC FRAMING
============================================================

Identify EVERY generic framing category that applies to
the article.

This is a MULTI-LABEL classification task.

You may select multiple labels.

You may select zero labels if none apply.

You MUST use the exact short label from the left side below.

Available framing labels:

{framing_options}

For example:

If the article uses an Economic frame and a Political frame,
return:

[
    "economic",
    "political"
]

Do NOT return the long descriptions.

Do NOT invent labels.

============================================================
TASK 2: POLITICAL LEANING
============================================================

Select EXACTLY ONE political leaning.

Available choices:

{leaning_options}

Do NOT invent a political leaning.

============================================================
REQUIRED OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly this format:

{{
    "framing_labels": [
        "economic",
        "political"
    ],
    "political_leaning": "left"
}}

Do not include explanations.
Do not include Markdown.
Do not put the JSON in a code block.

============================================================
ARTICLE
============================================================

{article_text}
""".strip()


# ============================================================
# LM STUDIO REQUEST
# ============================================================

def ask_lm_studio(article_text, MODEL_NAME, base_url):

    prompt = build_prompt(article_text)

    payload = {
        "model": MODEL_NAME,

        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise multi-label classification "
                    "system. Follow the requested JSON format exactly."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        "temperature": 0.0,

        # NOTE: deliberately NOT setting "response_format" here.
        # LM Studio only reliably supports response_format={"type": "json_schema", ...}
        # with a full schema attached; the simpler {"type": "json_object"} shorthand
        # gets rejected with a 400 by many backends/models. The prompt already
        # instructs the model to return raw JSON, and parse_model_response()
        # below can extract a JSON object even if the model wraps it in text,
        # so we don't need response_format to work correctly.
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Lost connection to LM Studio at {base_url} mid-run. "
            f"Is the server still running? ({e})"
        )

    if not response.ok:
        # Surface LM Studio's actual error message instead of a bare
        # "400 Client Error: Bad Request" with no explanation.
        raise RuntimeError(
            f"LM Studio returned {response.status_code} {response.reason} "
            f"for {base_url}/v1/chat/completions.\n"
            f"Response body: {response.text}"
        )

    result = response.json()

    content = result["choices"][0]["message"]["content"]

    return parse_model_response(content)


# ============================================================
# MODEL RESPONSE PARSING
# ============================================================

def parse_model_response(content):

    content = content.strip()

    # Try direct JSON first.
    try:
        return json.loads(content)

    except json.JSONDecodeError:
        pass

    # If the model surrounded its answer with extra text,
    # attempt to extract the JSON object.
    match = re.search(
        r"\{.*\}",
        content,
        flags=re.DOTALL,
    )

    if match:

        try:
            return json.loads(match.group(0))

        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not parse LM Studio response as JSON:\n"
        + content
    )


# ============================================================
# VALIDATE FRAMING
# ============================================================

def validate_frames(predicted):

    if predicted is None:
        return []

    if not isinstance(predicted, list):
        predicted = [predicted]

    valid = []

    for label in predicted:

        label = normalize(label)

        if label in VALID_FRAME_LABELS:
            valid.append(label)

        else:
            print(
                f"WARNING: Model returned invalid "
                f"framing label: {label}"
            )

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(valid))


# ============================================================
# VALIDATE POLITICAL LEANING
# ============================================================

def validate_political_leaning(predicted):

    if predicted is None:
        return None

    predicted = normalize(predicted)

    for leaning in POLITICAL_LEANINGS:

        if predicted == normalize(leaning):
            return leaning

    print(
        f"WARNING: Model returned invalid political "
        f"leaning: {predicted}"
    )

    return None


# ============================================================
# GRADE FRAMING
# ============================================================

def grade_framing(predicted, actual):

    predicted_set = set(
        normalize(x)
        for x in predicted
    )

    actual_set = set(
        normalize(x)
        for x in actual
    )

    correct = predicted_set & actual_set

    correct_count = len(correct)

    total_actual = len(actual_set)

    total_predicted = len(predicted_set)

    # This is the primary score you requested:
    #
    # number of correct labels / total ground-truth labels

    label_accuracy = (
        correct_count / total_actual
        if total_actual > 0
        else 0.0
    )

    precision = (
        correct_count / total_predicted
        if total_predicted > 0
        else 0.0
    )

    recall = (
        correct_count / total_actual
        if total_actual > 0
        else 0.0
    )

    if precision + recall > 0:

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    exact_match = (
        predicted_set == actual_set
    )

    return {
        "correct_labels": correct_count,

        "total_ground_truth_labels": total_actual,

        "predicted_labels": total_predicted,

        "label_accuracy": label_accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "exact_match": exact_match,

        "correct_label_names": sorted(correct_set := correct),
    }


# ============================================================
# GRADE POLITICAL LEANING
# ============================================================

def grade_political_leaning(predicted, actual):

    if predicted is None or actual is None:

        return {
            "correct": False,
            "scored": False,
        }

    correct = (
        normalize(predicted)
        == normalize(actual)
    )

    return {
        "correct": correct,
        "scored": True,
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(
    articles_file,
    framing_file,
    output_file,
    MODEL_NAME,
    base_url=None,
):

    base_url = base_url or lm_studio_base_url()

    loaded_ids = check_lm_studio_connection(base_url)

    if not MODEL_NAME or MODEL_NAME == "local-model":
        # No specific model requested -- try to auto-detect.
        if len(loaded_ids) == 1:
            MODEL_NAME = loaded_ids[0]
            print(f"No model specified -- auto-detected the one loaded model: '{MODEL_NAME}'\n")
        elif len(loaded_ids) == 0:
            raise SystemExit(
                "\nNo model is loaded in LM Studio, so there's nothing to "
                "auto-detect. Load a model in LM Studio and try again.\n"
            )
        else:
            raise SystemExit(
                f"\n{len(loaded_ids)} models are loaded in LM Studio "
                f"({', '.join(loaded_ids)}), so I can't guess which one you "
                "want. Pick one explicitly with --model, or evaluate all "
                f"of them with --models \"{','.join(loaded_ids)}\".\n"
            )
    elif loaded_ids and MODEL_NAME not in loaded_ids:
        print(
            f"WARNING: '{MODEL_NAME}' is not among LM Studio's currently "
            f"loaded models ({', '.join(loaded_ids)}). The request may "
            "fail or silently hit the wrong model.\n"
        )

    # Always tag the output filename with the (now-resolved) model name,
    # so results are never ambiguous about which model produced them and
    # never silently overwrite results from a different model.
    output_file = model_tagged_filename(output_file, MODEL_NAME)

    print("Loading articles.json...")

    articles = load_articles(
        articles_file
    )

    print(
        f"Loaded {len(articles)} articles."
    )

    print(
        "Loading framing.jsonl..."
    )

    framing_records = load_framing_jsonl(
        framing_file
    )

    print(
        f"Loaded {len(framing_records)} framing records."
    )

    ground_truth_frames = build_ground_truth(
        framing_records
    )

    # --------------------------------------------------------
    # Determine which articles have matching framing labels.
    # --------------------------------------------------------

    matching_uuids = [
        uuid
        for uuid in ground_truth_frames
        if uuid in articles
    ]

    print(
        f"Found {len(matching_uuids)} articles with "
        "matching framing annotations."
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    total_correct_frames = 0
    total_ground_truth_frames = 0
    total_predicted_frames = 0

    total_exact_matches = 0

    political_correct = 0
    political_scored = 0

    results = []

    # --------------------------------------------------------
    # Process each article
    # --------------------------------------------------------

    for index, uuid in enumerate(matching_uuids):

        print()
        print("=" * 70)
        print(
            f"ARTICLE {index + 1}/{len(matching_uuids)}"
        )
        print(f"UUID: {uuid}")

        article = articles[uuid]

        article_text = article.get("text")

        if not article_text:

            print("ERROR: Article has no text.")

            results.append({
                "uuid": uuid,
                "error": "No article text",
            })

            continue

        # Ground truth framing.
        actual_frames = ground_truth_frames[uuid]

        # Ground truth political leaning comes from
        # articles.json.
        actual_leaning = article.get(
            "political_leaning"
        )

        print(
            "Ground-truth frames:",
            actual_frames
        )

        print(
            "Ground-truth political leaning:",
            actual_leaning
        )

        # ----------------------------------------------------
        # Ask LM Studio
        # ----------------------------------------------------

        print("Sending article to LM Studio...")

        try:

            prediction = ask_lm_studio(
                article_text,
                MODEL_NAME,
                base_url,
            )

        except Exception as e:

            print(
                "ERROR communicating with LM Studio:"
            )

            print(e)

            results.append({
                "uuid": uuid,
                "error": str(e),
            })

            continue

        # ----------------------------------------------------
        # Parse model response
        # ----------------------------------------------------

        predicted_frames = validate_frames(
            prediction.get("framing_labels")
        )

        predicted_leaning = validate_political_leaning(
            prediction.get("political_leaning")
        )

        print(
            "Predicted frames:",
            predicted_frames
        )

        print(
            "Predicted political leaning:",
            predicted_leaning
        )

        # ----------------------------------------------------
        # Grade
        # ----------------------------------------------------

        framing_score = grade_framing(
            predicted_frames,
            actual_frames,
        )

        political_score = grade_political_leaning(
            predicted_leaning,
            actual_leaning,
        )

        # ----------------------------------------------------
        # Accumulate framing metrics
        # ----------------------------------------------------

        total_correct_frames += (
            framing_score["correct_labels"]
        )

        total_ground_truth_frames += (
            framing_score[
                "total_ground_truth_labels"
            ]
        )

        total_predicted_frames += (
            framing_score[
                "predicted_labels"
            ]
        )

        if framing_score["exact_match"]:
            total_exact_matches += 1

        # ----------------------------------------------------
        # Accumulate political metrics
        # ----------------------------------------------------

        if political_score["scored"]:

            political_scored += 1

            if political_score["correct"]:

                political_correct += 1

        # ----------------------------------------------------
        # Save individual result
        # ----------------------------------------------------

        result = {

            "uuid": uuid,

            "title": article.get(
                "title"
            ),

            "source_domain": article.get(
                "source_domain"
            ),

            # -----------------------------
            # Framing
            # -----------------------------

            "actual_framing_labels":
                actual_frames,

            "predicted_framing_labels":
                predicted_frames,

            "correct_framing_labels":
                framing_score[
                    "correct_labels"
                ],

            "total_ground_truth_framing_labels":
                framing_score[
                    "total_ground_truth_labels"
                ],

            "framing_label_accuracy":
                framing_score[
                    "label_accuracy"
                ],

            "framing_precision":
                framing_score[
                    "precision"
                ],

            "framing_recall":
                framing_score[
                    "recall"
                ],

            "framing_f1":
                framing_score[
                    "f1"
                ],

            "framing_exact_match":
                framing_score[
                    "exact_match"
                ],

            "correct_framing_label_names":
                framing_score[
                    "correct_label_names"
                ],

            # -----------------------------
            # Political leaning
            # -----------------------------

            "actual_political_leaning":
                actual_leaning,

            "predicted_political_leaning":
                predicted_leaning,

            "political_leaning_correct":
                political_score[
                    "correct"
                ],

            "political_leaning_scored":
                political_score[
                    "scored"
                ],
        }

        results.append(result)

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print()
        print(
            "FRAMING SCORE:"
        )

        print(
            f"  {framing_score['correct_labels']} / "
            f"{framing_score['total_ground_truth_labels']}"
        )

        print(
            f"  Label accuracy: "
            f"{framing_score['label_accuracy']:.4f}"
        )

        print(
            f"  Exact match: "
            f"{framing_score['exact_match']}"
        )

        print()
        print(
            "POLITICAL LEANING:"
        )

        print(
            f"  {predicted_leaning} "
            f"vs "
            f"{actual_leaning}"
        )

        print(
            f"  Correct: "
            f"{political_score['correct']}"
        )

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    overall_framing_accuracy = (
        total_correct_frames
        / total_ground_truth_frames
        if total_ground_truth_frames > 0
        else 0.0
    )

    overall_precision = (
        total_correct_frames
        / total_predicted_frames
        if total_predicted_frames > 0
        else 0.0
    )

    overall_recall = (
        total_correct_frames
        / total_ground_truth_frames
        if total_ground_truth_frames > 0
        else 0.0
    )

    if overall_precision + overall_recall > 0:

        overall_f1 = (
            2
            * overall_precision
            * overall_recall
            / (
                overall_precision
                + overall_recall
            )
        )

    else:

        overall_f1 = 0.0

    exact_match_accuracy = (
        total_exact_matches
        / len(results)
        if results
        else 0.0
    )

    political_accuracy = (
        political_correct
        / political_scored
        if political_scored > 0
        else 0.0
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output = {

        "configuration": {

            "model": MODEL_NAME,

            "framing_labels": FRAMING_LABELS,

            "political_leanings":
                POLITICAL_LEANINGS,
        },

        "summary": {

            "articles_processed":
                len(results),

            # -----------------------------------------------
            # Primary framing metric
            # -----------------------------------------------

            "framing_correct_labels":
                total_correct_frames,

            "framing_total_ground_truth_labels":
                total_ground_truth_frames,

            "framing_label_accuracy":
                overall_framing_accuracy,

            # -----------------------------------------------
            # Additional framing metrics
            # -----------------------------------------------

            "framing_precision":
                overall_precision,

            "framing_recall":
                overall_recall,

            "framing_f1":
                overall_f1,

            "framing_exact_match_count":
                total_exact_matches,

            "framing_exact_match_accuracy":
                exact_match_accuracy,

            # -----------------------------------------------
            # Political leaning
            # -----------------------------------------------

            "political_leaning_correct":
                political_correct,

            "political_leaning_scored":
                political_scored,

            "political_leaning_accuracy":
                political_accuracy,
        },

        "individual_results":
            results,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print()
    print("FRAMING")
    print("-" * 70)

    print(
        f"Correct labels: "
        f"{total_correct_frames} / "
        f"{total_ground_truth_frames}"
    )

    print(
        f"Label accuracy: "
        f"{overall_framing_accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{overall_precision:.4f}"
    )

    print(
        f"Recall: "
        f"{overall_recall:.4f}"
    )

    print(
        f"F1: "
        f"{overall_f1:.4f}"
    )

    print(
        f"Exact match: "
        f"{total_exact_matches} / "
        f"{len(results)}"
    )

    print()
    print("POLITICAL LEANING")
    print("-" * 70)

    print(
        f"Correct: "
        f"{political_correct} / "
        f"{political_scored}"
    )

    print(
        f"Accuracy: "
        f"{political_accuracy:.4f}"
    )

    print()
    print(
        f"Results saved to: {output_file}"
    )

    return output


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an LM Studio model on "
            "the MM-Framing dataset."
        )
    )

    parser.add_argument(
        "--articles",
        default="articles.json",
        help="Path to articles.json",
    )

    parser.add_argument(
        "--framing",
        default="framing.jsonl",
        help="Path to framing.jsonl",
    )

    parser.add_argument(
        "--output",
        default="mm_framing_results.json",
        help="Output JSON file",
    )

    parser.add_argument(
        "--model",
        default="local-model",
        help=(
            "LM Studio model name (single model). Leave unset to "
            "auto-detect -- if exactly one model is loaded in LM Studio, "
            "it will be used automatically; if zero or multiple are "
            "loaded, you'll be told to specify one explicitly."
        ),
    )

    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated list of LM Studio model IDs to evaluate and "
            "compare in one run, e.g. --models \"qwen2.5-7b-instruct,"
            "llama-3.1-8b-instruct\". Load all of them in LM Studio first "
            "(Multi Model Session, or just load them one at a time -- "
            "LM Studio keeps multiple models resident simultaneously). "
            "Overrides --model. Each model gets its own output file."
        ),
    )

    parser.add_argument(
        "--host",
        default=LM_STUDIO_HOST,
        help=(
            "Host where LM Studio's server is running (default: localhost). "
            "Change this if this script runs on a different machine/VM/"
            "container than LM Studio itself."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=LM_STUDIO_PORT,
        help="Port LM Studio's local server is listening on (default: 1234)",
    )

    args = parser.parse_args()

    base_url = lm_studio_base_url(args.host, args.port)

    model_list = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else [args.model]
    )

    all_outputs = {}

    for model_name in model_list:
        print("\n" + "#" * 70)
        print(f"# EVALUATING MODEL: {model_name}")
        print("#" * 70 + "\n")

        # evaluate() tags the output filename with the model name itself
        # (resolving auto-detection first if model_name is blank), so we
        # just pass the base path through unchanged here.
        result = evaluate(
            articles_file=args.articles,
            framing_file=args.framing,
            output_file=args.output,
            MODEL_NAME=model_name,
            base_url=base_url,
        )
        all_outputs[model_name] = result

    if len(model_list) > 1:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        header = f"{'Model':<35} {'Frame F1':>10} {'Frame Recall':>13} {'Leaning Acc':>12}"
        print(header)
        print("-" * len(header))
        for model_name, result in all_outputs.items():
            s = result["summary"]
            print(
                f"{model_name:<35} "
                f"{s['framing_f1']:>10.4f} "
                f"{s['framing_recall']:>13.4f} "
                f"{s['political_leaning_accuracy']:>12.4f}"
            )


if __name__ == "__main__":
    main()
