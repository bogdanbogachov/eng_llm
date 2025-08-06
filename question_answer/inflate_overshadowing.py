import json
from typing import List, Dict
from collections import Counter


def sort_json_by_title_and_answer(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as infile:
        data: List[Dict] = json.load(infile)

    # Count how often each title appears
    title_counts = Counter(item.get("title", "") for item in data)

    # Sort by: (title frequency, title alphabetically, answer alphabetically)
    sorted_data = sorted(
        data,
        key=lambda item: (
            title_counts[item.get("title", "")],
            item.get("title", ""),
            item.get("answer", "")
        )
    )

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(sorted_data, outfile, indent=4)

    return None


def inflate_qa_answers_with_file_inputs(qa_original_path: str, inflating_path: str, qa_output_path: str):
    # Sort the QAs by title and answer
    sort_json_by_title_and_answer(qa_original_path, qa_output_path)

    # Load data from files
    with open(qa_output_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    with open(inflating_path, "r", encoding="utf-8") as f:
        inflating_data = json.load(f)

    inflating_texts = [item["text"] for item in inflating_data]
    inflating_count = len(inflating_texts)

    # Group QAs by title
    title_to_qas = {}
    for entry in qa_data:
        title = entry.get("title", "")
        title_to_qas.setdefault(title, []).append(entry)

    # Assign inflation per distinct answer per title
    for title in sorted(title_to_qas.keys()):
        answer_to_inflation = {}
        inflating_index = 0

        for qa in title_to_qas[title]:
            answer_text = qa["answer"]

            if answer_text not in answer_to_inflation:
                if inflating_index >= inflating_count:
                    raise IndexError(f"Not enough inflating texts for title: {title}")
                answer_to_inflation[answer_text] = inflating_texts[inflating_index]
                inflating_index += 1

            inflating_text = answer_to_inflation[answer_text]
            qa["answer"] = f"{inflating_text}\n\n{answer_text}"

    # Flatten back into a list
    result = [qa for qas in title_to_qas.values() for qa in qas]

    with open(qa_output_path, "w", encoding="utf-8") as outfile:
        json.dump(result, outfile, indent=4)

    return None
