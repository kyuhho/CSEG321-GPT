# quantize_model.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
torch.backends.quantized.engine = 'qnnpack'  # ✅ 필수 설정

from models.gpt2 import GPT2Model
from config import GPT2Config

# quantize_model.py 수정
def load_student_model(checkpoint_path: str) -> GPT2Model:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    config_dict = ckpt["config"]
    config = GPT2Config(**config_dict)        # 동적 config 로드
    model = GPT2Model(config)                 # 구조를 맞춰 초기화
    model.load_state_dict(ckpt["state_dict"]) # 정상 로딩
    model.eval()
    return model

def quantize_model(model: torch.nn.Module) -> torch.nn.Module:
    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

def save_model(model: torch.nn.Module, path: str):
    torch.save(model.state_dict(), path)  # 또는 torch.save(model, path) 로 전체 저장 가능

if __name__ == "__main__":
    student_ckpt_path = "models/student.pt"
    quant_ckpt_path = "student_quant.pt"

    print("📦 Loading student model...")
    model = load_student_model(student_ckpt_path)

    print("⚙️  Applying quantization...")
    quant_model = quantize_model(model)

    print("💾 Saving quantized model...")
    save_model(quant_model, quant_ckpt_path)

    print(f"✅ Quantized model saved to {quant_ckpt_path}")