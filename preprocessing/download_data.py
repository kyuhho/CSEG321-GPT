# pip install datasets 실행해야함
from datasets import load_dataset
import json
import os

# Hugging Face에서 CNN/DailyMail 데이터셋 로드
dataset = load_dataset("cnn_dailymail", "3.0.0")

# 저장 폴더 만들기
os.makedirs("summarization/data", exist_ok=True)

# 데이터 저장 (100개만 예시로 저장)
for split in ["train", "validation", "test"]:
    subset = dataset[split]
    data = [{"article": item["article"], "summary": item["highlights"]} for item in subset]

    with open(f"summarization/data/{split}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print("데이터 저장 완료")
