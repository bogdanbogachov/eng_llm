from question_answer import populate
from evaluate import load_data, evaluate
from inference import ask_engineering_question


if __name__ == '__main__':
    # # Question-answers
    # populate(file_to_read='question_answer/srm.pdf')

    # # Evaluate
    # predictions_file = "answers/test.json"  # Replace with your file path
    # ground_truth_file = "question_answer/qa_pairs_test.json"  # Replace with your file path
    # predictions, ground_truth = load_data(predictions_file, ground_truth_file)
    # results = evaluate(predictions, ground_truth)

    # Infer
    test = ask_engineering_question("What is the formula for bending stress?")
    print(test)
