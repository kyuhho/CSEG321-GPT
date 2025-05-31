import os
import random
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from models.gpt2 import GPT2Model as StudentGPT2Model
from config import GPT2Config
from datasets import load_dataset
import argparse


def seed_everything(seed: int = 11711):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_student_model(checkpoint_path: str, device: torch.device) -> StudentGPT2Model:
    """
    저장된 체크포인트에서 Student GPT-2 모델을 로드합니다.
    """
    # 학습 시 사용한 것과 동일한 GPT2Config
    student_config = GPT2Config(
        vocab_size=50260,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=1536
    )
    student_model = StudentGPT2Model(student_config)
    student_model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    student_model.load_state_dict(ckpt["model_state_dict"])
    student_model.eval()
    return student_model


class CNNDailyMailSampleDataset(Dataset):
    """
    테스트(split="test")에서 임의로 선택된 idx 리스트를 바탕으로
    Article만 뽑아 "Article: ...\nSummary:" 형식의 입력 시퀀스를 만들어 반환합니다.
    """
    def __init__(self, hf_dataset, tokenizer: GPT2Tokenizer, indices: list, max_length: int = 512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.indices = indices
        self.max_length = max_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 실제 데이터셋의 인덱스를 뽑아서 사용
        real_idx = self.indices[idx]
        example = self.dataset[real_idx]
        article = example["article"]
        gold_summary = example["highlights"]

        # "Article: ...\nSummary:" 프롬프트 생성
        prompt = "Article: " + article.strip() + "\nSummary:"
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),         # (seq_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),# (seq_len,)
            "article": article,
            "gold_summary": gold_summary
        }


def generate_summary_greedy(student_model: StudentGPT2Model,
                            tokenizer: GPT2Tokenizer,
                            input_ids: torch.LongTensor,
                            attention_mask: torch.LongTensor,
                            device: torch.device,
                            max_new_tokens: int = 100) -> str:
    """
    Student GPT-2 모델로 Greedy 방식 요약 생성.
    - input_ids: 이미 "[Article: ...\nSummary:]" 토큰화된 텐서 (1D)
    - attention_mask: 이 때의 어텐션 마스크 (1D)
    - max_new_tokens: 프롬프트 뒤에 생성할 최대 토큰 수
    """
    student_model.eval()
    # (1) 배치 차원 추가
    generated = input_ids.unsqueeze(0).to(device)        # (1, seq_len)
    attn_mask = attention_mask.unsqueeze(0).to(device)   # (1, seq_len)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = student_model(input_ids=generated, attention_mask=attn_mask)
            hidden = outputs["last_hidden_state"]  # (1, cur_len, hidden_size)
            logits = student_model.hidden_state_to_token(hidden)  # (1, cur_len, vocab_size)
            next_token_logits = logits[:, -1, :]     # (1, vocab_size)
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (1,1)

            # EOS 토큰이 나오면 생성 중단
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # 새 토큰을 뒤에 붙이고 어텐션 마스크도 늘리기
            generated = torch.cat([generated, next_token_id], dim=1)  # (1, cur_len+1)
            attn_mask = torch.cat([attn_mask, torch.ones((1, 1), dtype=torch.long).to(device)], dim=1)

    # 프롬프트 길이만큼 자르고, 나머지 디코딩
    prompt_len = input_ids.size(0)
    gen_ids = generated[0, prompt_len:]  # (gen_len,)
    summary = tokenizer.decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    return summary.strip()


def test_inference_random(args):
    # (0) 시드 설정 & 디바이스 확인
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # (1) Teacher 계열과 동일한 토크나이저 로드
    tokenizer = GPT2Tokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
    tokenizer.pad_token = tokenizer.eos_token

    # (2) Student 모델 로드
    student_model = load_student_model(args.checkpoint, device)

    # (3) 테스트 데이터셋 로드
    hf_test = load_dataset("abisee/cnn_dailymail", "3.0.0")["test"]
    total_test = len(hf_test)
    print(f"Total test examples available: {total_test}")

    # (4) 무작위로 indices 뽑기
    #     args.num_to_sample 만큼 랜덤하게 선택
    if args.num_to_sample > total_test:
        raise ValueError(f"Requested sample size {args.num_to_sample} > total test examples {total_test}")
    random_indices = random.sample(range(total_test), args.num_to_sample)
    print(f"Randomly selected indices (0-based): {random_indices}\n")

    # (5) 샘플 전용 Dataset/Dataloader (batch_size=1로 처리)
    sample_dataset = CNNDailyMailSampleDataset(hf_test, tokenizer, random_indices, max_length=args.max_length)

    # (6) 각 샘플에 대해 요약 생성 및 출력
    print(f"--- Generating {args.num_to_sample} random summaries using distilled student model ---\n")
    for idx_in_list in range(len(sample_dataset)):
        item = sample_dataset[idx_in_list]
        prompt_ids = item["input_ids"]         # (seq_len,)
        prompt_mask = item["attention_mask"]    # (seq_len,)
        article_text = item["article"]
        gold_summary = item["gold_summary"]

        pred_summary = generate_summary_greedy(
            student_model, tokenizer,
            prompt_ids, prompt_mask,
            device,
            max_new_tokens=args.gen_max_length
        )

        print(f"[Sample {idx_in_list + 1} / Original Index {random_indices[idx_in_list]}]")
        print("Article:   ", article_text[:200].replace("\n", " "), "...")
        print("Gold Summ: ", gold_summary)
        print("Pred Summ: ", pred_summary)
        print("-" * 80)

    print("\n>>> Inference (random 3) complete.")


def get_args():
    parser = argparse.ArgumentParser(description="Random 3 Inference with Distilled GPT-2")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the student checkpoint (e.g. models/student.pt)"
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Max tokenization length for 'Article: ... \\nSummary:'"
    )
    parser.add_argument(
        "--num_to_sample", type=int, default=3,
        help="Number of random test examples to run inference on (default=3)"
    )
    parser.add_argument(
        "--gen_max_length", type=int, default=150,
        help="Max number of new tokens to generate per summary"
    )
    parser.add_argument(
        "--seed", type=int, default=11711,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    test_inference_random(args)
