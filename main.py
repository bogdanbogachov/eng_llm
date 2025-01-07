from question_answer import populate
from evaluate import load_data, evaluate
from inference import ask_engineering_question
import argparse


if __name__ == '__main__':
    # Arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_qa", type=bool, default=False)
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--infer", type=bool, default=False)
    args = parser.parse_args()

    # Question-answers
    if args.create_qa:
        populate(file_to_read='question_answer/srm.pdf')

    # Evaluate
    if args.evaluate:
        predictions_file = "answers/test.json"  # Replace with your file path
        ground_truth_file = "question_answer/qa_pairs_test.json"  # Replace with your file path
        predictions, ground_truth = load_data(predictions_file, ground_truth_file)
        results = evaluate(predictions, ground_truth)

    # Infer
    if args.infer:
        test = ask_engineering_question("What is the formula for bending stress?")
        print(test)
