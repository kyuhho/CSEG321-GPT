# python summary_generation.py --model_type baseline 로 base line 성능테스트
# python summary_generation.py --model_type ours 로 경량화 한 모델 성능테스트

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel
)
import psutil
import os
import gc
import tracemalloc

from config import GPT2Config  # 필요 시 사용
from tqdm import tqdm
import evaluate
import os
import sys
import json

# 모델 경로 추가
sys.path.append('distillation')
sys.path.append('.')

def get_model_memory_usage():
    """현재 모델이 사용하는 GPU/CPU 메모리 측정"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    else:
        # CPU의 경우 tracemalloc 사용
        current, peak = tracemalloc.get_traced_memory()
        return current / (1024 ** 2)  # MB

def measure_inference_memory(model, tokenizer, article, model_type):
    """추론 시 메모리 사용량을 정확히 측정하는 함수"""
    # 메모리 측정 시작
    if not torch.cuda.is_available():
        tracemalloc.start()
    
    # 가비지 컬렉션으로 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 추론 전 메모리
    memory_before = get_model_memory_usage()
    
    # 실제 추론 수행
    summary = generate_summary(model, tokenizer, article, model_type)
    
    # 추론 후 메모리
    memory_after = get_model_memory_usage()
    
    # 메모리 측정 종료
    if not torch.cuda.is_available():
        tracemalloc.stop()
    
    inference_memory = memory_after - memory_before
    return summary, max(inference_memory, 0)  # 음수 방지

def load_quantized_model(checkpoint_path: str):
    print(f"📦 Loading quantized model from {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model = ckpt['model']
        config = ckpt['config']
        print(f"✅ Loaded quantized model with config:")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Num layers: {config.num_hidden_layers}")
        print(f"  - Num attention heads: {config.num_attention_heads}")
        model.eval()
        return model, config
    except Exception as e:
        print(f"❌ Error loading quantized model: {e}")
        raise e

def load_model(model_type):
    # 모델 로딩 전 메모리
    memory_before_loading = get_model_memory_usage()
    
    if model_type == "baseline":
        print("📦 Loading baseline model: gavin124/gpt2-finetuned-cnn-summarization-v2")
        tokenizer = GPT2Tokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
        model.eval()
        
    elif model_type == "ours":
        print("📦 Loading our quantized GPT2 model")
        tokenizer = GPT2Tokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model, config = load_quantized_model("saved_models/student_quant.pt")
        
    else:
        raise ValueError("Unknown model type. Choose from ['baseline', 'ours'].")
    
    # 모델 로딩 후 메모리
    memory_after_loading = get_model_memory_usage()
    model_memory = memory_after_loading - memory_before_loading
    
    print(f"📊 Model memory usage: {model_memory:.2f} MB")
    
    return model, tokenizer, model_memory

def generate_summary(model, tokenizer, article, model_type):
    max_article_length = 800
    if len(article) > max_article_length:
        article = article[:max_article_length]

    prompt = f"Article: {article.strip()}\nSummary:"

    try:
        if model_type == "baseline":
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=False)
            input_length = inputs["input_ids"].shape[1]
            if input_length > 400:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300, padding=False)

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
        else:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=False)
            with torch.no_grad():
                input_ids = inputs["input_ids"]
                generated_ids = input_ids.clone()
                max_new_tokens = 100

                for _ in range(max_new_tokens):
                    attention_mask = torch.ones_like(generated_ids)
                    model_outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
                    hidden_states = model_outputs['last_hidden_state']
                    logits = model.hidden_state_to_token(hidden_states)
                    next_token_logits = logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                outputs = generated_ids

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Summary:" in generated_text:
            summary = generated_text.split("Summary:")[-1].strip()
        else:
            input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            summary = generated_text.replace(input_text, "").strip() if input_text in generated_text else generated_text.strip()
        return summary

    except Exception as e:
        print(f"Error generating summary: {e}")
        import traceback
        traceback.print_exc()
        return "Error: Could not generate summary"

def main(args):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # ✅ 반드시 CPU 강제 사용
    device = torch.device("cpu")
    print(f"\n🚀 Evaluating model: {args.model_type}")
    print(f"🔧 Using device: {device} (forced CPU for quantized models)\n")

    try:
        model, tokenizer, model_memory = load_model(args.model_type)
        print("📊 Loading dataset...")
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")["test"]
        dataset = dataset.select(range(args.num_samples))

        predictions, references, summaries_to_save, inference_memories = [], [], [], []
        successful_generations = 0

        for i, item in enumerate(tqdm(dataset, desc="📝 Generating summaries")):
            article = item["article"]
            reference = item["highlights"]
            
            # 정확한 메모리 측정으로 추론
            summary, inference_memory = measure_inference_memory(model, tokenizer, article, args.model_type)
            
            if summary and not summary.startswith("Error:"):
                inference_memories.append(inference_memory)
                
                predictions.append(summary)
                references.append(reference)
                summaries_to_save.append({
                    "id": i,
                    "article": article[:300] + "...",
                    "reference": reference,
                    "summary": summary,
                    "inference_memory_mb": inference_memory
                })
                successful_generations += 1
            else:
                print(f"Failed to generate summary for sample {i}")

        print(f"\n✅ Successfully generated {successful_generations}/{args.num_samples} summaries")

        if predictions:
            print("📊 Computing ROUGE scores...")
            rouge = evaluate.load("rouge")
            scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
            print("\n📊 ROUGE Scores:")
            for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
                if key in scores:
                    print(f"{key.upper()} - F1: {scores[key]:.4f}")

            output_file = f"summaries_{args.model_type}_{args.num_samples}.json"
            avg_inference_memory = sum(inference_memories) / len(inference_memories) if inference_memories else 0.0
            rouge_l_score = round(scores.get("rougeL", 0.0), 4)

            print(f"\n📊 Memory Usage Summary:")
            print(f"  - Model loading memory: {model_memory:.2f} MB")
            print(f"  - Average inference memory per sample: {avg_inference_memory:.2f} MB")
            print(f"  - Total memory (model + avg inference): {model_memory + avg_inference_memory:.2f} MB")

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "model_name": args.model_type,
                    "scores": scores,
                    "rouge_l": rouge_l_score,
                    "model_memory_mb": round(model_memory, 2),
                    "avg_inference_memory_mb": round(avg_inference_memory, 2),
                    "total_memory_mb": round(model_memory + avg_inference_memory, 2),
                    "summaries": summaries_to_save
                }, f, indent=2, ensure_ascii=False)

            print(f"\n💾 All results saved to {output_file}")
            
            # 평가 결과 별도 저장
            eval_result = {
                "model_name": args.model_type,
                "rouge_l": rouge_l_score,
                "model_memory_mb": round(model_memory, 2),
                "avg_inference_memory_mb": round(avg_inference_memory, 2),
                "total_memory_mb": round(model_memory + avg_inference_memory, 2)
            }

            with open(f"evaluation_result_{args.model_type}.json", "w") as f:
                json.dump(eval_result, f, indent=2)

            print(f"\n✅ Evaluation result saved to evaluation_result_{args.model_type}.json")
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