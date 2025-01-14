from config import CONFIG
from question_answer import populate
from evaluate import load_data, evaluate
from inference import ask_slg, ask_baseline, ask_baseline_finetuned, AskRag
import argparse


if __name__ == '__main__':
    # Arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_qa", type=bool, default=False)
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--infer_slg", type=bool, default=False)
    parser.add_argument("--infer_baseline", type=bool, default=False)
    args = parser.parse_args()

    # Question-answers
    if args.create_qa:
        populate(file_to_read='question_answer/srm.pdf')

    # Infer baseline
    if args.infer_baseline:
        ask_baseline(file='question_answer/qa_pairs_test.json')
        ask_baseline_finetuned(file='question_answer/qa_pairs_test.json')
        rag = AskRag(
            documents_file='question_answer/qa_pairs_test.json',
            questions_file='question_answer/qa_pairs_test.json')
        rag.generate_responses()

    # Infer slg
    if args.infer_slg:
        ask_slg(file='question_answer/qa_pairs_test.json', inference_model='slg')

    # Evaluate
    if args.evaluate:
        predictions_file = "answers/test.json"  # Replace with your file path
        ground_truth_file = "question_answer/qa_pairs_test.json"  # Replace with your file path
        predictions, ground_truth = load_data(predictions_file, ground_truth_file)
        results = evaluate(predictions, ground_truth)
