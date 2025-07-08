import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from huggingface_hub import login
import os
import re
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from transformers import pipeline

from logging_config import logger
from config import CONFIG


# Download WordNet for synonym matching
nltk.download('wordnet')

logger.propagate = False

def load_data(predictions_file, ground_truth_file):
    """
    Load predictions and ground truth answers from JSON files.
    """
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)

    return predictions, ground_truth


def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score between a reference and a candidate answer.
    """
    reference_tokens = reference.split()  # BLEU expects a list of references
    candidate_tokens = candidate.split()
    smoothing = SmoothingFunction().method4  # For smoothing BLEU scores
    return sentence_bleu(references=reference_tokens, hypothesis=candidate_tokens, smoothing_function=smoothing)


def calculate_rouge(reference, candidate):
    """
    Calculate ROUGE scores between a reference and a candidate answer.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(target=reference, prediction=candidate)
    return scores


def calculate_exact_match(reference, candidate):
    """
    Calculate Exact Match (EM) score between a reference and a candidate answer.
    """
    return int(reference.strip() == candidate.strip())


def calculate_ai_expert(reference, candidate):
    """
    Calculate AI expert scores between a reference and a candidate answer.
    """
    api_key = CONFIG['api_key']
    model_id = CONFIG['3_3_70b']
    ai_expert_prompt = CONFIG['ai_expert_prompt']
    query_expert_prompt = CONFIG["query_expert_prompt"]
    max_new_tokens = CONFIG['max_new_tokens']
    seed = CONFIG['seed']
    temperature = CONFIG['temperature']
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    login(api_key)
    client = InferenceClient()
    message = [
        {"role": "system", "content": ai_expert_prompt},
        {"role": "user", "content": query_expert_prompt.format(text_1=reference, text_2=candidate)}
    ]

    # apply the chat template to the messages
    total = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    llm_response = client.text_generation(
        total,
        model=model_id,
        max_new_tokens=max_new_tokens,
        seed=seed,
        temperature=temperature
    )

    try:
        return int(llm_response)
    except Exception:
        return 0


# Function to calculate METEOR scores
def calculate_meteor_score(reference, candidate):
    """
    Calculate meteor score between a reference and a candidate answer.
    """
    score = meteor_score(references=[word_tokenize(reference)], hypothesis=word_tokenize(candidate))
    return score


def check_entailment(reference: str, candidate: str, nli_model) -> int:
    """
    Check if the hypothesis (generated answer) is entailed by the premise (reference answer).

    Returns:
        1 if ENTALIMENT is the top prediction, 0 otherwise.
    """
    result = nli_model({
        "reference": reference,
        "candidate": candidate
    })[0]

    return int(result['label'] == "ENTAILMENT")


def evaluate(predictions, ground_truth):
    """
    Evaluate predictions against ground truth using BLEU, ROUGE, and Exact Match.
    """

    # Load once and reuse
    nli_model = pipeline("text-classification", model="roberta-large-mnli")

    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    exact_matches = []
    ai_experts = []
    meteor_scores = []
    bert_scores = []
    logger.info(f"Evaluation has started.")
    counter = 0
    threshold = 10
    for pred, truth in zip(predictions, ground_truth):
        # if counter in [i for i in range(0, len(ground_truth), 30)]:
        #     time.sleep(45)
        counter += 1
        percent_checked = (counter/len(predictions))*100
        if threshold < percent_checked:
            logger.info(f"Evaluated {threshold}%")
            threshold += 10
        if pred['question'] != truth['question']:
            logger.info(f"Warning: questions didn't match at chapter: {pred['chapter']}, and title: {pred['title']}")
            logger.info(f"Pred question: {pred['question']}")
            logger.info(f"Truth question: {truth['question']}")
            logger.info(40*'-')
            continue

        if truth['answer'] == '':
            gt_answer = 'Empty'
        elif truth['answer'] is None:
            gt_answer = 'Empty'
        else:
            gt_answer = truth['answer']

        if pred['answer'] == '':
            pred_answer = 'Empty'
        elif pred['answer'] is None:
            pred_answer = 'Empty'
        else:
            pred_answer = pred['answer']

        # BLEU
        logger.debug("Calculating BLEU score.")
        bleu = calculate_bleu(gt_answer, pred_answer)
        bleu_scores.append(bleu)

        # ROUGE
        logger.debug("Calculating ROUGE score.")
        rouge = calculate_rouge(gt_answer, pred_answer)
        for key in rouge_scores.keys():
            rouge_scores[key].append(rouge[key].fmeasure)

        # Exact Match
        logger.debug("Calculating Exact Match (EM) score.")
        exact_matches.append(calculate_exact_match(gt_answer, pred_answer))

        # AI expert
        logger.debug("Calculating AI expert score.")
        ai_expert = calculate_ai_expert(gt_answer, pred_answer)
        ai_experts.append(ai_expert)

        # Meteor
        logger.debug("Calculating METEOR score.")
        meteor = calculate_meteor_score(gt_answer, pred_answer)
        meteor_scores.append(meteor)

        # BERT
        logger.debug("Calculating BERT score.")
        bert = check_entailment(gt_answer, pred_answer, nli_model=nli_model)
        bert_scores.append(bert)

    # Aggregate scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge = {key: sum(rouge_scores[key]) / len(rouge_scores[key]) for key in rouge_scores.keys()}
    exact_match_score = sum(exact_matches) / len(exact_matches) if exact_matches else 0
    # ai_expert_score = sum(ai_experts) / len(ai_experts) if ai_experts else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    logger.info(f"Evaluation has been completed.")
    logger.info(40*'-')
    # time.sleep(30)

    return {
        "BLEU": avg_bleu,
        "ROUGE": avg_rouge,
        "Exact Match": exact_match_score,
        # "AI Expert": ai_expert_score
        "METEOR": avg_meteor
    }


def extract_log_values(log_file):
    with open(log_file, 'r') as f:
        content = f.read()
        content = re.sub(r'\bnan\b', "100000", content)
        content = re.sub(r'\binf\b', "100000", content)
        data = json.loads(content.replace("'", '"'))

        second_last_log = data[-2]  # Get the second-to-last dictionary
        last_log = data[-1]  # Get the last dictionary

        return {
            'train_loss': second_last_log['train_loss'],
            'train_runtime': second_last_log['train_runtime'],
            'eval_loss': last_log['eval_loss'],
            'epochs': last_log['epoch']
        }


def pull_training_metrics(base_folder):
    metrics = []
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if folder == 'slg':
            all_train_loss = 0
            all_eval_loss = 0
            total_train_runtime = 0
            total_epochs = 0
            count = 0

            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    training_log_path = os.path.join(subfolder_path, 'training_log.txt')
                    if os.path.exists(training_log_path):
                        logger.debug(training_log_path)
                        log_values = extract_log_values(training_log_path)
                        all_train_loss += log_values['train_loss']
                        all_eval_loss += log_values['eval_loss']
                        total_train_runtime += log_values['train_runtime']
                        total_epochs += log_values['epochs']
                        count += 1


            metrics.append(
                {
                    'avg_train_loss_slg': all_train_loss / count,
                    'avg_eval_loss_slg': all_eval_loss / count,
                    'total_train_runtime_slg': total_train_runtime,
                    'avg_epochs_slg': total_epochs / count
                }
            )
        elif folder != 'slg' and 'logs' not in folder:
            training_log_path = os.path.join(folder_path, 'training_log.txt')
            logger.debug(training_log_path)
            if os.path.exists(training_log_path):
                log_values = extract_log_values(training_log_path)
                metrics.append(
                    {
                        f'train_loss_{folder}': log_values['train_loss'],
                        f'all_eval_loss_{folder}': log_values['eval_loss'],
                        f'train_runtime_{folder}': log_values['train_runtime'],
                        f'epochs_{folder}': log_values['epochs']
                    }
                )

    return metrics
