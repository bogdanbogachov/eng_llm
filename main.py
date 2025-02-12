from config import CONFIG

import argparse
import json
import os


if __name__ == '__main__':
    # Arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_qa", type=bool, default=False)
    parser.add_argument("--split_qa", type=bool, default=False)
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--infer_slg", type=bool, default=False)
    parser.add_argument("--infer_baseline", type=bool, default=False)
    parser.add_argument("--infer_finetuned", type=bool, default=False)
    parser.add_argument("--download_models", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=False)
    args = parser.parse_args()

    # Download models
    if args.download_models:
        from download_llama import download, download_llama_3_1_8b
        download(model_name=CONFIG['3_2_1b'], save_directory='downloaded_3_2_1b')
        download_llama_3_1_8b(model_name=CONFIG['3_1_8b'], save_directory='downloaded_3_1_8b')

    # Question-answers
    if args.create_qa:
        from question_answer import populate
        populate(file_to_read='question_answer/srm.pdf')

    if args.split_qa:
        from question_answer import split_qa_pairs_by_title, split_train_test
        split_train_test('question_answer/qa_pairs.json')
        split_qa_pairs_by_title('question_answer/qa_train.json')

    # Experiments
    for experiment in [2]:
        # Finetune
        if args.finetune:
            from finetune import finetune
            os.makedirs('experiments', exist_ok=True)
            # Finetune SLG
            for file in os.listdir("question_answer/split_by_title"):
                finetune(
                    model_to_tune='downloaded_3_2_1b',
                    adapter_name=os.path.splitext(file)[0],
                    data=f"question_answer/split_by_title/{file}",
                    experiment_number=experiment,
                    slg=True
                )

            # Finetune an orchestrator for SLG
            finetune(
                model_to_tune='downloaded_3_1_8b',
                adapter_name='orchestrator_3_1_8b',
                data='question_answer/qa_train.json',
                experiment_number=experiment,
                orchestrator=True
            )

            # Finetune llama 3.2 1B instruct
            finetune(
                model_to_tune='downloaded_3_2_1b',
                adapter_name='3_2_1b',
                data='question_answer/qa_train.json',
                experiment_number=experiment
            )

            # Finetune llama 3.1 8B instruct
            finetune(
                model_to_tune='downloaded_3_1_8b',
                adapter_name='3_1_8b',
                data='question_answer/qa_train.json',
                experiment_number=experiment
            )

        # Infer baseline
        if args.infer_baseline:
            from inference import ask_baseline, AskRag
            os.makedirs(f'answers/{experiment}', exist_ok=True)
            ask_baseline(file='question_answer/qa_test.json', model=CONFIG['3_2_1b'], experiment=experiment)
            ask_baseline(file='question_answer/qa_test.json', model=CONFIG['3_1_8b'], experiment=experiment)
            ask_baseline(file='question_answer/qa_test.json', model=CONFIG['3_3_70b'], experiment=experiment)
            rag = AskRag(
                documents_file='question_answer/qa_train.json',
                questions_file='question_answer/qa_test.json',
                experiment=experiment)
            rag.generate_responses()

        if args.infer_finetuned:
            from inference import ask_finetuned
            os.makedirs(f'answers/{experiment}', exist_ok=True)
            ask_finetuned(file='question_answer/qa_test.json',
                          base_model='downloaded_3_2_1b',
                          adapter=f'experiments/{experiment}/finetuned_3_2_1b',
                          experiment=experiment)
            ask_finetuned(file='question_answer/qa_test.json',
                          base_model='downloaded_3_1_8b',
                          adapter=f'experiments/{experiment}/finetuned_3_1_8b',
                          experiment=experiment)

        # Infer slg
        if args.infer_slg:
            from inference import SmallLanguageGraph
            os.makedirs(f'answers/{experiment}', exist_ok=True)
            slg = SmallLanguageGraph(experts_location=experiment, experiment=experiment)
            slg.ask_slg(
                file='question_answer/qa_test.json'
            )

        # Evaluate
        if args.evaluate:
            from evaluate import load_data, evaluate
            metrics_list = []
            ground_truth_file = "question_answer/qa_test.json"
            for predictions_file in os.listdir(f"answers/{experiment}"):
                predictions, ground_truth = load_data(f'answers/{experiment}/{predictions_file}',
                                                      ground_truth_file)
                results = evaluate(predictions, ground_truth)
                new_dict = {os.path.splitext(predictions_file)[0]: results}
                metrics_list.append(new_dict)
                with open(f'experiments/{experiment}/metrics.json', 'w') as f:
                    json.dump(metrics_list, f, indent=4)
