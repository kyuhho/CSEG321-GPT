# python summary_generation.py --model_type baseline 로 base line 성능테스트
# python summary_generation.py --model_type ours 로 경량화 한 모델 성능테스트

import torch
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GPT2Tokenizer
)

from config import GPT2Config  # 필요 시 사용

def load_model(model_type, device):
    if model_type == "baseline":
        print("\U0001F4E6 Loading baseline model: facebook/bart-large-cnn")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)
        return model, tokenizer

    elif model_type == "ours":
        print("\U0001F4E6 Loading our custom GPT2 model")
        from models.gpt2 import GPT2ModelForGeneration  # 너의 모델 이름에 맞게 수정
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2ModelForGeneration.from_pretrained("path_to_your_model").to(device)
        return model, tokenizer

    else:
        raise ValueError("Unknown model type. Choose from ['baseline', 'ours'].")

def generate_summary(model, tokenizer, article, device, model_type):
    if model_type == "baseline":
        inputs = tokenizer(
            article, return_tensors="pt",
            truncation=True, padding=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)

    else:  # ours
        inputs = tokenizer(
            "Article: " + article.strip() + "\nSummary:",
            return_tensors="pt",
            truncation=True, padding="max_length",
            max_length=1024
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id
            )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Evaluating model: {args.model_type}\n")

    # 1. 모델 로드
    model, tokenizer = load_model(args.model_type, device)

    # 2. CNN/DailyMail 데이터 로드
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")["test"]
    dataset = dataset.select(range(args.num_samples))  # 일부 샘플만 사용

    # 3. 요약 생성
    predictions = []
    references = []
    summaries_to_save = []

    for item in tqdm(dataset, desc="📝 Generating summaries"):
        article = item["article"]
        reference = item["highlights"]
        summary = generate_summary(model, tokenizer, article, device, args.model_type)

        predictions.append(summary)
        references.append(reference)
        summaries_to_save.append({
            "article": article[:300] + "...",
            "reference": reference,
            "summary": summary
        })

    # 4. ROUGE 평가
    rouge = load_metric("rouge")
    scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

    print("\n📊 ROUGE Scores:")
    for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        if key in scores:
            print(f"{key.upper()} - F1: {scores[key]:.4f}")

    # 5. 요약 저장
    output_path = f"generated_summaries_{args.model_type}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries_to_save, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Summaries saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["baseline", "ours"], required=True)
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples to evaluate")
    args = parser.parse_args()

    main(args)
