# python summary_generation.py --model_type baseline 로 base line 성능테스트
# python summary_generation.py --model_type ours 로 경량화 한 모델 성능테스트

import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GPT2Tokenizer
)

# from models.gpt2 import GPT2WithLMHead  # 필요 시 주석 해제
from config import GPT2Config  # 필요 시 실제 config 사용

def load_model(model_type, device):
    if model_type == "baseline":
        print("\U0001F4E6 Loading baseline model: facebook/bart-large-cnn")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)
        return model, tokenizer

    elif model_type == "ours":
        print("\U0001F4E6 Loading our custom GPT2 model")
        # ✅ 여기 직접 정의한 모델 로딩 코드로 수정하세요
        from models.gpt2 import GPT2ModelForGeneration  # 예시용, 실제 클래스명에 맞게 수정
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2ModelForGeneration.from_pretrained("path_to_your_model").to(device)
        return model, tokenizer

    else:
        raise ValueError("Unknown model type. Choose from ['baseline', 'ours'].")

def summarize_baseline(model, tokenizer, test_data, device):
    summaries = []
    for item in tqdm(test_data, desc="\U0001F4DD Summarizing with baseline model"):
        inputs = tokenizer(
            item['article'], return_tensors="pt",
            truncation=True, padding=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append({"summary": summary})  # ✅ id 제거
    return summaries

def summarize_ours(model, tokenizer, test_data, device):
    summaries = []
    for item in tqdm(test_data, desc="\U0001F4DD Summarizing with our custom GPT2 model"):
        inputs = tokenizer(
            item['article'], return_tensors="pt",
            truncation=True, padding="max_length",
            max_length=1024
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id
            )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append({"summary": summary})  # ✅ id 제거
    return summaries

def load_test_data(path):
    test_path = os.path.join(path, "test.json")
    with open(test_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_summaries(summaries, path):
    output_path = os.path.join(path, "generated_summaries.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\n\U0001F4C4 Summaries saved to {output_path}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Evaluating model: {args.model_type}\n")

    model, tokenizer = load_model(args.model_type, device)
    test_data = load_test_data(args.data_dir)

    if args.model_type == "baseline":
        summaries = summarize_baseline(model, tokenizer, test_data, device)
    else:
        summaries = summarize_ours(model, tokenizer, test_data, device)

    save_summaries(summaries, args.data_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["baseline", "ours"], required=True)
    parser.add_argument("--data_dir", type=str, default="data/cnndata")
    args = parser.parse_args()

    main(args)
