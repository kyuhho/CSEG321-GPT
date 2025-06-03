import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from tqdm import tqdm
import evaluate  # evaluate 라이브러리로 변경
import os
import sys

# 모델 경로 추가
sys.path.append('distillation')
sys.path.append('.')

def load_quantized_model(checkpoint_path: str, device):
    """Quantized 모델을 로드하는 함수"""
    from models.gpt2 import GPT2Model
    from config import GPT2Config
    
    # Quantized 모델 로드
    print(f"📦 Loading quantized model from {checkpoint_path}")
    
    # 원본 student 모델 구조를 먼저 로드
    student_path = "saved_models/student.pt"
    student_ckpt = torch.load(student_path, map_location='cpu', weights_only=False)
    config = student_ckpt["config"]
    
    # Student 모델 생성
    model = GPT2Model(config)
    
    # Quantized 상태 로드
    quantized_state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(quantized_state)
    
    # Dynamic quantization 적용
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    model = model.to(device)
    model.eval()
    
    return model, config

def load_model(model_type, device):
    if model_type == "baseline":
        print("📦 Loading baseline model: gavin124/gpt2-finetuned-cnn-summarization-v2")
        tokenizer = GPT2Tokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2").to(device)
        
        # 모델과 토크나이저 호환성 확인
        print(f"Model vocab size: {model.config.vocab_size}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        
        return model, tokenizer

    elif model_type == "ours":
        print("📦 Loading our quantized GPT2 model")
        
        # Tokenizer는 baseline과 동일하게 사용
        tokenizer = GPT2Tokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Quantized 모델 로드
        model, config = load_quantized_model("saved_models/student_quant.pt", device)
        
        print(f"✅ Loaded quantized model with config:")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Num layers: {config.num_hidden_layers}")
        print(f"  - Num attention heads: {config.num_attention_heads}")
        
        return model, tokenizer

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

        else:  # ours - quantized model
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            ).to(device)

            # 우리 모델은 custom GPT2Model이므로 직접 forward pass 수행
            with torch.no_grad():
                # 입력 길이 확인
                input_length = inputs["input_ids"].shape[1]
                max_new_tokens = 100
                
                input_ids = inputs["input_ids"]
                generated_ids = input_ids.clone()
                
                # Auto-regressive generation
                for _ in range(max_new_tokens):
                    # 현재까지 생성된 토큰들로 다음 토큰 예측
                    model_outputs = model(input_ids=generated_ids)
                    hidden_states = model_outputs['last_hidden_state']
                    
                    # Hidden states를 logits로 변환
                    logits = model.hidden_state_to_token(hidden_states)
                    
                    # 마지막 토큰의 logits만 사용
                    next_token_logits = logits[:, -1, :]
                    
                    # 다음 토큰 선택 (greedy decoding)
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # EOS 토큰이면 생성 중단
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    
                    # 생성된 토큰을 시퀀스에 추가
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                outputs = generated_ids

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
        import traceback
        traceback.print_exc()
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