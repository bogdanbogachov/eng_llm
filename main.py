from config import CONFIG
from question_answer import populate, split_qa_pairs_by_title, split_train_test
from evaluate import load_data, evaluate
from inference import SmallLanguageGraph, ask_baseline, ask_baseline_finetuned, AskRag
from download_llama import download
from finetune import finetune_slg_node
import argparse
import json
import os


if __name__ == '__main__':
    # Arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_qa", type=bool, default=False)
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--infer_slg", type=bool, default=False)
    parser.add_argument("--infer_baseline", type=bool, default=False)
    parser.add_argument("--download_models", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=False)
    args = parser.parse_args()

    # Download models
    if args.download_models:
        download("meta-llama/Llama-3.2-1B-Instruct", "local_llama_model_1")

    # Question-answers
    if args.create_qa:
        populate(file_to_read='question_answer/srm.pdf')
        split_train_test('question_answer/qa_pairs.json')
        split_qa_pairs_by_title('question_answer/qa_train.json')

    for experiment in [1]:
        # Finetune
        if args.finetune:
            os.makedirs('experiments', exist_ok=True)
            for file in os.listdir("question_answer/split_by_title"):
                finetune_slg_node(
                    node=os.path.splitext(file)[0],
                    data=f"question_answer/split_by_title/{file}",
                    experiment_number=experiment)

        # Infer baseline
        if args.infer_baseline:
            os.makedirs('answers', exist_ok=True)
            ask_baseline(file='question_answer/qa_test.json')
            ask_baseline_finetuned(file='question_answer/qa_test.json')
            rag = AskRag(
                documents_file='question_answer/qa_train.json',
                questions_file='question_answer/qa_test.json')
            rag.generate_responses()

        # Infer slg
        if args.infer_slg:
            os.makedirs('answers', exist_ok=True)
            slg = SmallLanguageGraph(experts_location=experiment)
            slg.ask_slg(
                file='question_answer/qa_test.json',
                inference_model='slg'
            )

        # Evaluate
        if args.evaluate:
            metrics_list = []
            ground_truth_file = "question_answer/qa_test.json"
            for predictions_file in os.listdir("answers"):
                predictions, ground_truth = load_data(f'answers/{predictions_file}', ground_truth_file)
                results = evaluate(predictions, ground_truth)
                new_dict = {os.path.splitext(predictions_file)[0]: results}
                metrics_list.append(new_dict)
                with open(f'experiments/{experiment}/metrics.json', 'w') as f:
                    json.dump(metrics_list, f, indent=4)
