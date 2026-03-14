from typing import List, Dict, Any
import re

def clean_answer(text: str) -> str:
    """
    Cleans the answer text from markup, symbols, and units.
    - Removes bold tags (**), non-breaking spaces, and quotes.
    - Converts commas to dots for decimal numbers.
    - Removes percentage signs and units like 'people', 'years', 'times', 'percent'.
    - If the text contains a number + unit pattern (e.g., '3.5 years'), it extracts only the number.
    - Retains non-numeric text such as names or cities.
    """
    if text is None:
        return ""

    text = text.strip().replace('**', '').replace('\u202f', '').replace('"', '')

    # Extract only the part after "Answer:" or "Jawaban:" if present
    text = re.sub(r'^[Jj]awaban\s*:\s*', '', text).strip()

    # Convert commas to dots for decimal numbers (e.g., 3,5 -> 3.5)
    text = re.sub(r'(\d+),(\d+)', r'\1.\2', text)

    # Pure number pattern (can contain decimals and specific Indonesian/English units)
    numeric_match = re.match(
        r'^\s*([+-]?\d+(?:\.\d+)?)\s*(?:%|persen|orang|tahun|kali lipat|kali|unit|buah)?\s*$',
        text, flags=re.IGNORECASE
    )
    if numeric_match:
        return numeric_match.group(1).strip()

    # If the text starts with a number followed by any word/unit -> extract only the number
    mixed_match = re.match(r'^\s*([+-]?\d+(?:\.\d+)?)(?:\s+\w+)+', text)
    if mixed_match:
        return mixed_match.group(1).strip()

   # Remove trailing percentage signs or dots if any
    text = re.sub(r'%$', '', text)
    if re.match(r'^\d+\.$', text):
        text = text[:-1]
    elif re.search(r'\.$', text) and not re.match(r'^\d+\.\d+$', text):
        text = re.sub(r'\.$', '', text)

    return text.strip()


def normalize_answer(text: str) -> str:
    """
    Normalize answer for comparison.
    - Convert to lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Remove extra whitespace
    """
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text


def exact_match(predicted: str, ground_truth: str) -> bool:
    """
    Calculate exact match after normalization.
    """
    return normalize_answer(predicted) == normalize_answer(ground_truth)


def f1_score(predicted: str, ground_truth: str) -> float:
    """
    Calculate token-level F1 score.
    """
    pred_tokens = normalize_answer(predicted).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    if len(common_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate evaluation metrics for all results.
    """
    total = len(results)
    exact_matches = 0
    f1_scores = []
    
    for result in results:
        predicted = result.get('extracted_answer', '')
        ground_truth = result.get('answer', '')
        
        # Exact match
        if exact_match(predicted, ground_truth):
            exact_matches += 1
        
        # F1 score
        f1 = f1_score(predicted, ground_truth)
        f1_scores.append(f1)
    
    metrics = {
        'total': total,
        'exact_match': exact_matches / total if total > 0 else 0.0,
        'f1_score': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }
    
    return metrics