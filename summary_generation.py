import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from datasets import load_dataset
from tqdm import tqdm
import evaluate  # evaluate 라이브러리로 변경
from models.gpt2 import GPT2Model as StudentGPT2Model
import os

def load_quantized_model(path, device='cpu'):
    """양자화된 모델 로드"""
    print(f"Loading quantized model from {path}")
    checkpoint = torch.load(path, map_location=device)
    
    quantized_model = checkpoint['quantized_model']
    config = checkpoint['config']
    
    return quantized_model, config

def load_model(model_type, device):
    if model_type == "baseline":
        print("📦 Loading baseline model: gavin124/gpt2-finetuned-cnn-summarization-v2")
        tokenizer = GPT2Tokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2").to(device)
        print(f"Model vocab size: {model.config.vocab_size}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        return model, tokenizer

    elif model_type == "ours":
        print("📦 Loading our quantized GPT2 model")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 양자화된 모델 로드
        quant_model, config = load_quantized_model("saved_models/student_quant.pt", device)
        quant_model.eval()
        
        # 양자화된 모델은 이미 CPU/GPU에 적절히 배치되어 있음
        return quant_model, tokenizer

    else:
        raise ValueError("Unknown model type. Choose from ['baseline', 'ours'].")

def generate_summary(model, tokenizer, article, device, model_type):
    # 기사 길이 제한 (너무 긴 입력 방지)
    max_article_length = 800
    if len(article) > max_article_length:
        article = article[:max_article_length]
    
    prompt = f"Article: {article.strip()}\nSummary:"
    
    try:
        if model_type == "baseline":
            # 패딩 없이 처리
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,  # 1024에서 512로 줄임
                padding=False    # 패딩 제거
            ).to(device)
            
            # 입력 길이 확인
            input_length = inputs["input_ids"].shape[1]
            print(f"Input tokens: {input_length}")
            
            # 입력이 너무 길면 더 줄이기
            if input_length > 400:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=300,
                    padding=False
                ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=100,  # 128에서 100으로 줄임
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    # attention_mask 제거 (패딩이 없으므로)
                )

        else:  # ours
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=100,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # "Summary:" 이후 부분만 추출
        if "Summary:" in generated_text:
            summary = generated_text.split("Summary:")[-1].strip()
        else:
            # 입력 부분 제거
            input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            if input_text in generated_text:
                summary = generated_text.replace(input_text, "").strip()
            else:
                summary = generated_text.strip()
        
        return summary
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Error: Could not generate summary"

def main(args):
    # CUDA 디버깅 활성화
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Evaluating model: {args.model_type}")
    print(f"🔧 Using device: {device}\n")

    try:
        # 1. 모델 로드
        model, tokenizer = load_model(args.model_type, device)
        
        # 2. CNN/DailyMail 데이터 로드
        print("📊 Loading dataset...")
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")["test"]
        dataset = dataset.select(range(args.num_samples))

        # 3. 요약 생성
        predictions = []
        references = []
        summaries_to_save = []
        
        successful_generations = 0
        
        for i, item in enumerate(tqdm(dataset, desc="📝 Generating summaries")):
            try:
                article = item["article"]
                reference = item["highlights"]
                summary = generate_summary(model, tokenizer, article, device, args.model_type)
                
                if summary and not summary.startswith("Error:"):
                    predictions.append(summary)
                    references.append(reference)
                    summaries_to_save.append({
                        "id": i,
                        "article": article[:300] + "...",
                        "reference": reference,
                        "summary": summary
                    })
                    successful_generations += 1
                else:
                    print(f"Failed to generate summary for sample {i}")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        print(f"\n✅ Successfully generated {successful_generations}/{args.num_samples} summaries")

        if len(predictions) > 0:
            # 4. ROUGE 평가
            print("📊 Computing ROUGE scores...")
            rouge = evaluate.load("rouge")  # load_metric 대신 evaluate.load 사용
            scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

            print("\n📊 ROUGE Scores:")
            for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
                if key in scores:
                    print(f"{key.upper()} - F1: {scores[key]:.4f}")
                    
            # 결과 저장
            import json
            output_file = f"summaries_{args.model_type}_{args.num_samples}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "scores": scores,
                    "summaries": summaries_to_save
                }, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to {output_file}")
            
        else:
            print("❌ No successful summaries generated!")

    except Exception as e:
        print(f"❌ Main execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["baseline", "ours"])
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()
    main(args)