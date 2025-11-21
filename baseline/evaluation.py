import re
import json
import nltk

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu as bleu # bleu score
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge # rouge score
from nltk.translate.meteor_score import meteor_score # meteor score
from evaluate import load # bert score

class Evaluator:
    def __init__(self, device):
        self.device = device
        
        # Bert score
        self.bert_metric = load("bertscore")

        # for Bleu
        self.smoothing = SmoothingFunction().method7

        # rouge score
        self.rouge_metric = Rouge()

    def compute_bleu(self, reference, candidate):
        scores = []
        for ref, cand in zip(reference, candidate):
            ref_tokens = word_tokenize(ref)
            cand_tokens = word_tokenize(cand)

            if not ref_tokens or not cand_tokens:
                scores.append(0)
                continue

            scores.append(bleu([ref_tokens], cand_tokens, smoothing_function = self.smoothing))

        return sum(scores) / len(scores) if scores else 0

    def compute_rouge(self, reference, candidate):
        score = self.rouge_metric.get_scores(candidate, reference, avg=True)
        return score

    def compute_meteor(self, reference, candidate):
        scores = []
        for ref, cand in zip(reference, candidate):
            ref_tokens = word_tokenize(ref.lower())
            cand_tokens = word_tokenize(cand.lower())
            scores.append(meteor_score([ref_tokens], cand_tokens))

        return sum(scores) / len(scores) if scores else 0

    def cider_score(self, reference, candidate):
        pass

    def bert_score(self, reference, candidate):
        results = self.bert_metric.compute(predictions=candidate,
                                    references=reference,
                                    lang="en",
                                    model_type = "roberta-large-mnli",
                                    device = self.device,
                                    idf = True)

        precision = sum(results["precision"]) / len(results["precision"])
        recall = sum(results["recall"]) / len(results["recall"])
        f1 = sum(results["f1"]) / len(results["f1"])

        return precision, recall, f1
    
    def compute_token(self, reference, candidate):
        correct_tokens = 0
        total_ref_tokens = 0
        total_cand_tokens = 0
        for ref, cand in zip(reference, candidate):
            ref_tokens = word_tokenize(ref.lower())
            cand_tokens = word_tokenize(cand.lower())

            for token in cand_tokens:
                if token in ref_tokens:
                    correct_tokens += 1
            total_cand_tokens += len(cand_tokens)
            total_ref_tokens += len(ref_tokens)

        precision = correct_tokens / total_cand_tokens if total_cand_tokens > 0 else 0
        recall = correct_tokens / total_ref_tokens if total_ref_tokens > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1
    
    def evaluate_all(self, reference, candidate):
        if not reference or not candidate:
            raise ValueError("Reference and candidate lists must not be empty.")
        
        bleu = self.compute_bleu(reference, candidate)
        rouge = self.compute_rouge(reference, candidate)
        meteor = self.compute_meteor(reference, candidate)
        bert_precision, bert_recall, bert_f1 = self.bert_score(reference, candidate)
        token_precision, token_recall, token_f1 = self.compute_token(reference, candidate)

        return {
            "bleu": bleu,
            "rouge": rouge,
            "meteor": meteor,
            "bert": {
                "precision": bert_precision,
                "recall": bert_recall,
                "f1": bert_f1
            },
            "token": {
                "precision": token_precision,
                "recall": token_recall,
                "f1": token_f1
            }
        }

def ExtractResponse(args, model_name : str, parameter : str) -> list:
    dir = f"./results/{model_name}_{parameter}_{args.seed}_prompt_{args.prompt}_results.json"
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

def extractResponse2(args, model_name):
    low_answer = []
    high_answer = []

    for resol in ["low", "high"]:
        dir = f"./results/{model_name}_{resol}_{args.seed}_prompt_{args.prompt}_results.json"

        if model_name == "geochat":
            with open(dir, "r") as f:
                response = [json.loads(line) for line in f]

        else:
            with open(dir, "r") as f:
                response = json.load(f)

        for res_dict in response:
            answer = res_dict['answer']

            if resol == "low":
                low_answer.append(answer)

            else:
                high_answer.append(answer)

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


def evaluation_main(args, device, test):
    low_answer, high_answer = ExtractResponse(args.model, str(args.param))
    low_reference, high_reference = ExtractReference(test)

    evaluation_functions = Evaluator(device)
    low_scores = evaluation_functions.evaluate_all(low_reference, low_answer)
    high_scores = evaluation_functions.evaluate_all(high_reference, high_answer)

    # print results
    print(f"{args.model} [Low Resolution] BERTScore - Precision: {low_scores['bert']['precision']:.4f}, Recall: {low_scores['bert']['recall']:.4f}, F1: {low_scores['bert']['f1']:.4f}")
    print(f"{args.model} [Low Resolution] BLEU Score: {low_scores['bleu']:.4f}")
    print(f"{args.model} [Low Resolution] ROUGE Score: {low_scores['rouge']}")
    print(f"{args.model} [Low Resolution] METEOR Score: {low_scores['meteor']}")
    print(f"{args.model} [Low Resolution] Token Score - Precision: {low_scores['token']['precision']:.4f}, Recall: {low_scores['token']['recall']:.4f}, F1: {low_scores['token']['f1']:.4f}")

    print(f"{args.model} [High Resolution] BERTScore - Precision: {high_scores['bert']['precision']:.4f}, Recall: {high_scores['bert']['recall']:.4f}, F1: {high_scores['bert']['f1']:.4f}")
    print(f"{args.model} [High Resolution] BLEU Score: {high_scores['bleu']:.4f}")
    print(f"{args.model} [High Resolution] ROUGE Score: {high_scores['rouge']}")
    print(f"{args.model} [High Resolution] METEOR Score: {high_scores['meteor']}")
    print(f"{args.model} [High Resolution] Token Score - Precision: {high_scores['token']['precision']:.4f}, Recall: {high_scores['token']['recall']:.4f}, F1: {high_scores['token']['f1']:.4f}")

def paper_model_evaluation_main(args, device, test):
    low_answer, high_answer = extractResponse2(args.model)
    low_reference, high_reference = ExtractReference(test)

    evaluation_functions = Evaluator(device)
    low_scores = evaluation_functions.evaluate_all(low_reference, low_answer)
    high_scores = evaluation_functions.evaluate_all(high_reference, high_answer)

    # print results
    print(f"{args.model} [Low Resolution] BERTScore - Precision: {low_scores['bert']['precision']:.4f}, Recall: {low_scores['bert']['recall']:.4f}, F1: {low_scores['bert']['f1']:.4f}")
    print(f"{args.model} [Low Resolution] BLEU Score: {low_scores['bleu']:.4f}")
    print(f"{args.model} [Low Resolution] ROUGE Score: {low_scores['rouge']}")
    print(f"{args.model} [Low Resolution] METEOR Score: {low_scores['meteor']}")
    print(f"{args.model} [Low Resolution] Token Score - Precision: {low_scores['token']['precision']:.4f}, Recall: {low_scores['token']['recall']:.4f}, F1: {low_scores['token']['f1']:.4f}")

    print(f"{args.model} [High Resolution] BERTScore - Precision: {high_scores['bert']['precision']:.4f}, Recall: {high_scores['bert']['recall']:.4f}, F1: {high_scores['bert']['f1']:.4f}")
    print(f"{args.model} [High Resolution] BLEU Score: {high_scores['bleu']:.4f}")
    print(f"{args.model} [High Resolution] ROUGE Score: {high_scores['rouge']}")
    print(f"{args.model} [High Resolution] METEOR Score: {high_scores['meteor']}")
    print(f"{args.model} [High Resolution] Token Score - Precision: {high_scores['token']['precision']:.4f}, Recall: {high_scores['token']['recall']:.4f}, F1: {high_scores['token']['f1']:.4f}")

def main() -> None:
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('wordnet')

if __name__ == "__main__":
    main()
