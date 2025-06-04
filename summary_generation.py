# python summary_generation.py --model_type baseline 로 base line 성능테스트
# python summary_generation.py --model_type ours 로 경량화 한 모델 성능테스트


import torch
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel
)
import psutil
import os

from config import GPT2Config  # 필요 시 사용

def load_model(model_type, device):
    if model_type == "baseline":
        print("\U0001F4E6 Loading baseline model: gavin124/gpt2-finetuned-cnn-summarization-v2")
        tokenizer = GPT2Tokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2").to(device)
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
        # GPT2 모델에 맞는 프롬프트 형식 사용
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
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

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

    # 입력 부분을 제거하고 새로 생성된 부분만 반환
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # "Summary:" 이후 부분만 추출
    if "Summary:" in generated_text:
        summary = generated_text.split("Summary:")[-1].strip()
    else:
        summary = generated_text.strip()
    
    return summary

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
    memory_usages = []

    for item in tqdm(dataset, desc="📝 Generating summaries"):
        article = item["article"]
        reference = item["highlights"]
        summary = generate_summary(model, tokenizer, article, device, args.model_type)


        # ✅ 메모리 사용량 측정 (CPU 기준)
        process = psutil.Process(os.getpid())
        mem_used = process.memory_info().rss / (1024 ** 2)  # MB 단위
        memory_usages.append(mem_used)

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
    #rouge_l = scores["rougeL"].mid.fmeasure if "rougeL" in scores else 0.0
    rouge_l = scores["rougeL"] if "rougeL" in scores else 0.0

    # 필요 시 요약 저장
    # # 5. 요약 저장
    # output_path = f"generated_summaries_{args.model_type}.json"
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(summaries_to_save, f, ensure_ascii=False, indent=2)
    # print(f"\n📄 Summaries saved to {output_path}")

    # 5. 메모리 사용량 저장
    avg_memory_usage = sum(memory_usages) / len(memory_usages)

    # 🔽 평가 결과 저장
    eval_result = {
        "model_name": args.model_type,
        "rouge_l": round(rouge_l, 4),
        "memory_usage_mb": round(avg_memory_usage, 2)
    }

    with open(f"evaluation_result_{args.model_type}.json", "w") as f:
        json.dump(eval_result, f, indent=2)

    print(f"\n✅ Evaluation result saved to evaluation_result_{args.model_type}.json")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["baseline", "ours"], required=True)
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples to evaluate")
    args = parser.parse_args()

    main(args)