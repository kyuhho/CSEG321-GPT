import os
import json
import torch
from transformers import GPT2Tokenizer
from tqdm import tqdm

# 경로 설정
input_dir = "data/cnndata"
output_dir = os.path.join(input_dir, "tokenized")
os.makedirs(output_dir, exist_ok=True)

# GPT2 tokenizer 불러오기
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2는 pad_token 없음

# 최대 길이 설정
max_input_length = 1024
max_summary_length = 128

def encode(example):
    input_enc = tokenizer(
        example["article"],
        truncation=True,
        padding="max_length",
        max_length=max_input_length,
        return_tensors="pt"
    )
    target_enc = tokenizer(
        example["summary"],
        truncation=True,
        padding="max_length",
        max_length=max_summary_length,
        return_tensors="pt"
    )

    return {
        "input_ids": input_enc["input_ids"].squeeze(0),
        "attention_mask": input_enc["attention_mask"].squeeze(0),
        "labels": target_enc["input_ids"].squeeze(0)
    }

# 각 split에서 처리
for split in ["train", "validation", "test"]:
    path = os.path.join(input_dir, f"{split}.json")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 테스트용: 100개만 처리
    print(f"{split} 원본 샘플 수: {len(raw_data)} → 100개만 토크나이즈")
    encoded_data = [encode(example) for example in tqdm(raw_data[:100])]

    # 전체 처리하려면 아래 주석 해제
    # print(f"{split} 전체 데이터 {len(raw_data)}개 토크나이징 중...")
    # encoded_data = [encode(example) for example in tqdm(raw_data)]

    torch.save(encoded_data, os.path.join(output_dir, f"{split}.pt"))
    print(f"✅ {split}.pt 저장 완료 ({len(encoded_data)}개 샘플)")
