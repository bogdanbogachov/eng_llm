import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from logging_config import setup_logger
import yaml
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from huggingface_hub import login
from sklearn.metrics import accuracy_score


# Load config parameters
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize the logger
logger = setup_logger()


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
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)


def calculate_rouge(reference, candidate):
    """
    Calculate ROUGE scores between a reference and a candidate answer.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
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
    api_key = config['api_key']
    model_id = config['model_id']
    ai_expert_prompt = config['ai_expert_prompt']
    query_expert_prompt = config["query_expert_prompt"]
    max_new_tokens = config['max_new_tokens']
    seed = config['seed']
    temperature = config['temperature']
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

    return int(llm_response)


def evaluate(predictions, ground_truth):
    """
    Evaluate predictions against ground truth using BLEU, ROUGE, and Exact Match.
    """
    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    exact_matches = []
    ai_experts = []

    for pred, truth in zip(predictions, ground_truth):
        if pred['question'] != truth['question']:
            logger.info(f"Warning: questions didn't match at chapter: {pred['chapter']}, and title: {pred['title']}")
            logger.info(f"Pred question: {pred['question']}")
            logger.info(f"Truth question: {truth['question']}")
            logger.info(40*'-')
            continue

        gt_answer = truth['answer']
        pred_answer = pred['answer']

        # BLEU
        bleu = calculate_bleu(gt_answer, pred_answer)
        bleu_scores.append(bleu)

        # ROUGE
        rouge = calculate_rouge(gt_answer, pred_answer)
        for key in rouge_scores.keys():
            rouge_scores[key].append(rouge[key].fmeasure)

        # Exact Match
        exact_matches.append(calculate_exact_match(gt_answer, pred_answer))

        # AI expert
        ai_expert = calculate_ai_expert(gt_answer, pred_answer)
        ai_experts.append(ai_expert)

    # Aggregate scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge = {key: sum(rouge_scores[key]) / len(rouge_scores[key]) for key in rouge_scores.keys()}
    exact_match_score = sum(exact_matches) / len(exact_matches) if exact_matches else 0
    ai_expert_score = sum(ai_experts) / len(ai_experts) if ai_experts else 0

    return {
        "BLEU": avg_bleu,
        "ROUGE": avg_rouge,
        "Exact Match": exact_match_score,
        "AI Expert": ai_expert_score
    }
