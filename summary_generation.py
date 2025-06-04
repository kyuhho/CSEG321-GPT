import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import os
import sys
import json

# 모델 경로 추가
sys.path.append('distillation')
sys.path.append('.')

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
    if model_type == "baseline":
        print("📦 Loading baseline model: gavin124/gpt2-finetuned-cnn-summarization-v2")
        tokenizer = GPT2Tokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
        model.eval()
        return model, tokenizer

    elif model_type == "ours":
        print("📦 Loading our quantized GPT2 model")
        tokenizer = GPT2Tokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model, config = load_quantized_model("saved_models/student_quant.pt")
        return model, tokenizer

    else:
        raise ValueError("Unknown model type. Choose from ['baseline', 'ours'].")

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
        model, tokenizer = load_model(args.model_type)
        print("📊 Loading dataset...")
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")["test"]
        dataset = dataset.select(range(args.num_samples))

        predictions, references, summaries_to_save = [], [], []
        successful_generations = 0

        for i, item in enumerate(tqdm(dataset, desc="📝 Generating summaries")):
            article = item["article"]
            reference = item["highlights"]
            summary = generate_summary(model, tokenizer, article, args.model_type)
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