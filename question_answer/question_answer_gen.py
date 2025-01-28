from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from huggingface_hub import login
from sklearn.model_selection import train_test_split

from .pdf_reader import read_doc
from logging_config import logger
from config import CONFIG

from collections import defaultdict
import json
import math
import os


def generate(text):
    """
        Generates questions for ground truth texts.
        Args:
            - number of questions to generate.
        Returns:
            - a tuple with all generated questions.
    """
    api_key = CONFIG['api_key']
    model_id = CONFIG['3_3_70b']
    system_prompt = CONFIG['system_prompt']
    query_prompt = CONFIG["query_prompt"]
    max_new_tokens = CONFIG['max_new_tokens']
    seed = CONFIG['seed']
    temperature = CONFIG['temperature']
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    login(api_key)
    client = InferenceClient()

    questions = tuple()
    # An average word has 4 letter, an average sentence has 20 words, thus I divide text by 80 to get the number
    # of sentences. The hypothesis behind is that each sentence should have a question.
    number_of_questions = max(1, math.ceil(int(len(text)/(4*20))))
    logger.info(f"Number of questions: {number_of_questions}")
    for i in range(0, number_of_questions):
        logger.info(f"Working on question # {i}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_prompt.format(text=text, questions=questions)}
        ]

        # apply the chat template to the messages
        total = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        llm_response = client.text_generation(
            total,
            model=model_id,
            max_new_tokens=max_new_tokens,
            seed=seed,
            temperature=temperature
        )
        questions += (llm_response, )

    return questions


def populate(file_to_read):
    df = read_doc(file_to_read)
    index_to_start_qa_gen = 0

    # Check if the file exists
    if not os.path.exists("question_answer/qa_pairs.json"):
        # If the file does not exist, initialize it as an empty list
        with open("question_answer/qa_pairs.json", "w") as json_file:
            json.dump([], json_file, indent=4)
    else:
        # Load JSON data
        json_file_path = "question_answer/qa_pairs.json"
        with open(json_file_path, "r") as file:
            data = json.load(file)  # Assumes the JSON file contains a list of dictionaries

        # Get the last dictionary
        last_dict = data[-1]

        # Specify the key whose value you want to compare
        key_to_compare = "answer"

        last_value = last_dict[key_to_compare]

        # Find the index of the last identical value in reverse order
        last_identical_index = len(data) - 1
        for i in range(len(data) - 2, -1, -1):  # Start from second-to-last element
            if data[i][key_to_compare] == last_value:
                last_identical_index = i
            else:
                break

        # Delete all these values in the JSON file
        data = data[:last_identical_index]

        # Save the updated JSON file
        with open(json_file_path, "w") as file:
            json.dump(data, file, indent=4)

        # Find the new last value in the last dictionary
        if len(data) > 0:
            last_dict = data[-1]  # Updated last dictionary
            new_last_value = last_dict[key_to_compare]

            # Find the last occurrence of the new_last_value in a specific column
            column_to_search = "text"
            index_to_start_qa_gen = df[df[column_to_search] == new_last_value].index[-1] + 1

    for index, row in df.iterrows():
        if index >= index_to_start_qa_gen:
            logger.info(f"Working on text # {index}")
            questions = generate(row['text'])
            for i, question in enumerate(questions):
                new_data = {
                    'chapter': row['chapter'],
                    'title': row['title'],
                    'question': question,
                    'answer': row['text']
                }

                # Now, open the file in read-write mode
                with open("question_answer/qa_pairs.json", "r+") as json_file:
                    # Load the existing data
                    try:
                        existing_data = json.load(json_file)
                    except json.JSONDecodeError:
                        existing_data = []  # Handle empty or invalid file

                    # Append the new dictionary
                    existing_data.append(new_data)

                    # Move to the beginning of the file and write updated data
                    json_file.seek(0)
                    json.dump(existing_data, json_file, indent=4)
                    json_file.truncate()  # Remove any leftover content

            logger.info(40*'-')

    return None


def split_train_test(data_file):
    # Load JSON file
    logger.info(f"Splitting {data_file} into train and test sets.")
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Split data into train and test sets (80% train, 20% test)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Save train and test data
    with open("question_answer/qa_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4)

    with open("question_answer/qa_test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)

    logger.info("Train and test splitting is complete.")

    return None


def split_qa_pairs_by_title(data_file):
    # Load the JSON data
    logger.info(f"Splitting {data_file} by title.")
    with open(data_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Group entries by unique "title"
    grouped_data = defaultdict(list)
    for entry in data:
        grouped_data[entry["title"]].append(entry)

    # Save each group as a separate JSON file
    for title, entries in grouped_data.items():
        # Create a valid filename by replacing spaces and special characters
        filename = f"{title.replace(' ', '_').replace('/', '_').lower()}.json"

        os.makedirs('question_answer/split_by_title', exist_ok=True)
        with open(f"question_answer/split_by_title/{filename}", "w", encoding="utf-8") as file:
            json.dump(entries, file, indent=4, ensure_ascii=False)

    logger.info(f"Title splitting is complete.")

    return None
