import re
import json
import nltk

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu as bleu # bleu score
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge # rouge score
from nltk.translate.meteor_score import meteor_score as meteor # meteor score
from evaluate import load # bert score

def ExtractResponse(model_name : str, parameter : str) -> list:
    dir = f"./results/{model_name}_{parameter}_results.json"
    with open(dir, "r") as f:
        response = json.load(f)
    
    low_answer = []
    high_answer = []
    if model_name == "qwen":
        pattern = r"\nassistant\n"

    elif model_name == "gemma":
        pattern = r"\nmodel\n"

    elif model_name == "blip2":
        pattern = r"\nAnswer:"

    for resolution, value in response.items():
        for batch in value:
            for item in batch:
                if resolution == "low":
                    low_answer.append(re.compile(pattern).split(item)[1].strip())
                elif resolution == "high":
                    high_answer.append(re.compile(pattern).split(item)[1].strip())

    return low_answer, high_answer

def ExtractReference(dataset : dict):
    low_reference = []
    high_reference = []
    for resolution, value in dataset.items():
        for question_id, item in value.items():
            if resolution == "low":
                low_reference.append(item["answer"])
            elif resolution == "high":
                high_reference.append(item["answer"])

    return low_reference, high_reference

def bleu_score(references : list, candidates : list):
    scores = []
    cc = SmoothingFunction()
    for ref, cand in zip(references, candidates):
        ref_tokens = word_tokenize(ref)
        cand_tokens = word_tokenize(cand)
        
        scores.append(bleu(ref_tokens, cand_tokens, smoothing_function = cc.method7))

    print(sum(scores))

    return sum(scores) / len(scores)

def rouge_score(references : list, candidates : list):
    rouge = Rouge()
    score = rouge.get_scores(candidates, references, avg=True)

    return score

def meteor_score(references : list, candidates : list):
    scores = []

def cider_score(references : list, candidates : list):
    pass

def bert_score(references : list, candidates : list, device):
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=candidates,
                                references=references,
                                lang="en",
                                model_type = "roberta-large-mnli",
                                device = device,
                                idf = True)

    precision = sum(results["precision"]) / len(results["precision"])
    recall = sum(results["recall"]) / len(results["recall"])
    f1 = sum(results["f1"]) / len(results["f1"])

    return precision, recall, f1

def main() -> None:
    nltk.download('punkt_tab')
    nltk.download('punkt')

if __name__ == "__main__":
    main()
