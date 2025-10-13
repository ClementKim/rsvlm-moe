import json
import nltk.translate.bleu_score as bleu

def ExtractResponse(dir : str) -> list:
    with open(dir, "r") as f:
        response = json.load(f)

    return answer

def main() -> None:
    pass

if __name__ == "__main__":
    main()