import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import os
import sys

# 모델 경로 추가
sys.path.append('distillation')
sys.path.append('.')

def load_tensorrt_model(checkpoint_path: str, device):
    """TensorRT 최적화된 모델을 로드하는 함수"""
    print(f"📦 Loading TensorRT optimized model from {checkpoint_path}")
    
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        model = ckpt['model']
        config = ckpt['config']
        optimization_type = ckpt.get('model_type', 'unknown')
        optimization_info = ckpt.get('optimization_info', {})
        
        print(f"✅ Loaded {optimization_type} model with config:")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Num layers: {config.num_hidden_layers}")
        print(f"  - Num attention heads: {config.num_attention_heads}")
        print(f"  - Optimization type: {optimization_type}")
        print(f"  - Device compatible: {optimization_info.get('device_compatible', 'unknown')}")
        
        model = model.to(device)
        model.eval()
        
        return model, config, optimization_type
        
    except Exception as e:
        print(f"❌ Error loading TensorRT model: {e}")
        raise e

def load_model(model_type, device):
    if model_type == "baseline":
        print("📦 Loading baseline model: gavin124/gpt2-finetuned-cnn-summarization-v2")
        tokenizer = GPT2Tokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2").to(device)
        
        print(f"Model vocab size: {model.config.vocab_size}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        
        return model, tokenizer, device

    elif model_type == "ours":
        print("📦 Loading our TensorRT optimized GPT2 model")
        
        # Tokenizer는 baseline과 동일하게 사용
        tokenizer = GPT2Tokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # TensorRT 최적화된 모델 로드
        model, config, optimization_type = load_tensorrt_model("saved_models/student_tensorrt.pt", device)
        
        return model, tokenizer, device, optimization_type

    else:
        raise ValueError("Unknown model type. Choose from ['baseline', 'ours'].")

def generate_summary_tensorrt(model, tokenizer, article, device, optimization_type):
    """TensorRT 최적화된 모델을 위한 summary generation"""
    max_article_length = 800
    if len(article) > max_article_length:
        article = article[:max_article_length]
    
    prompt = f"Article: {article.strip()}\nSummary:"
    
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding='max_length'  # TensorRT는 고정 크기 입력을 선호
        ).to(device)

        with torch.no_grad():
            input_length = inputs["input_ids"].shape[1]
            max_new_tokens = 100
            
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            generated_ids = input_ids.clone()
            generated_mask = attention_mask.clone()
            
            # Auto-regressive generation
            for step in range(max_new_tokens):
                # 현재 시퀀스의 유효한 길이 계산
                current_length = generated_ids.shape[1]
                
                if current_length >= 512:  # 최대 길이 도달
                    break
                
                # 모델 forward pass
                if optimization_type == "tensorrt":
                    # TensorRT 모델은 고정 크기 입력이 필요할 수 있음
                    model_outputs = model(input_ids=generated_ids, attention_mask=generated_mask)
                else:
                    # Fake quantization 모델
                    model_outputs = model(input_ids=generated_ids, attention_mask=generated_mask)
                
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
                
                # Attention mask 업데이트
                new_mask = torch.ones_like(next_token)
                generated_mask = torch.cat([generated_mask, new_mask], dim=-1)
            
            outputs = generated_ids

        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # "Summary:" 이후 부분만 추출
        if "Summary:" in generated_text:
            summary = generated_text.split("Summary:")[-1].strip()
        else:
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

def generate_summary(model, tokenizer, article, device, model_type, optimization_type=None):
    """Summary generation dispatcher"""
    if model_type == "baseline":
        return generate_summary_baseline(model, tokenizer, article, device)
    else:
        return generate_summary_tensorrt(model, tokenizer, article, device, optimization_type)

def generate_summary_baseline(model, tokenizer, article, device):
    """Baseline 모델을 위한 summary generation"""
    max_article_length = 800
    if len(article) > max_article_length:
        article = article[:max_article_length]
    
    prompt = f"Article: {article.strip()}\nSummary:"
    
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        ).to(device)
        
        input_length = inputs["input_ids"].shape[1]
        print(f"Input tokens: {input_length}")
        
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
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Summary:" in generated_text:
            summary = generated_text.split("Summary:")[-1].strip()
        else:
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
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Evaluating model: {args.model_type}")
    print(f"🔧 Using device: {device}\n")

    try:
        # 1. 모델 로드
        if args.model_type == "ours":
            model, tokenizer, actual_device, optimization_type = load_model(args.model_type, device)
            print(f"📌 Model running on: {actual_device}")
            print(f"🔧 Optimization type: {optimization_type}")
        else:
            model, tokenizer, actual_device = load_model(args.model_type, device)
            optimization_type = None
            print(f"📌 Model running on: {actual_device}")
        
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
                summary = generate_summary(model, tokenizer, article, actual_device, args.model_type, optimization_type)
                
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
                    print(f"✅ Generated summary {i+1}: {summary[:100]}...")
                else:
                    print(f"Failed to generate summary for sample {i}")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        print(f"\n✅ Successfully generated {successful_generations}/{args.num_samples} summaries")

        if len(predictions) > 0:
            # 4. ROUGE 평가
            print("📊 Computing ROUGE scores...")
            rouge = evaluate.load("rouge")
            scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

            print("\n📊 ROUGE Scores:")
            for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
                if key in scores:
                    print(f"{key.upper()} - F1: {scores[key]:.4f}")
                    
            # 결과 저장
            import json
            suffix = f"_{optimization_type}" if optimization_type else ""
            output_file = f"summaries_{args.model_type}{suffix}_{args.num_samples}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "scores": scores,
                    "summaries": summaries_to_save,
                    "optimization_type": optimization_type
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