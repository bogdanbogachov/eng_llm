import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from logging_config import logger
from config import CONFIG
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from huggingface_hub import login

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
    api_key = CONFIG['api_key']
    model_id = CONFIG['model_id']
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
        logger.info("Calculating BLEU score.")
        bleu = calculate_bleu(gt_answer, pred_answer)
        bleu_scores.append(bleu)

        # ROUGE
        logger.info("Calculating ROUGE score.")
        rouge = calculate_rouge(gt_answer, pred_answer)
        for key in rouge_scores.keys():
            rouge_scores[key].append(rouge[key].fmeasure)

        # Exact Match
        logger.info("Calculating Exact Match (EM) score.")
        exact_matches.append(calculate_exact_match(gt_answer, pred_answer))

        # AI expert
        logger.info("Calculating AI expert score.")
        ai_expert = calculate_ai_expert(gt_answer, pred_answer)
        ai_experts.append(ai_expert)

    # Aggregate scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge = {key: sum(rouge_scores[key]) / len(rouge_scores[key]) for key in rouge_scores.keys()}
    exact_match_score = sum(exact_matches) / len(exact_matches) if exact_matches else 0
    ai_expert_score = sum(ai_experts) / len(ai_experts) if ai_experts else 0
    logger.info(f"Evaluation has been completed.")
    logger.info(40*'-')

    return {
        "BLEU": avg_bleu,
        "ROUGE": avg_rouge,
        "Exact Match": exact_match_score,
        "AI Expert": ai_expert_score
    }
