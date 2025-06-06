# quantize_model.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
torch.backends.quantized.engine = 'qnnpack'  # ✅ 필수 설정

from models.gpt2 import GPT2Model
from config import GPT2Config

torch.serialization.add_safe_globals([GPT2Config])

def load_student_model(checkpoint_path: str) -> GPT2Model:
    """Student 모델을 로드하는 함수"""
    print(f"📦 Loading student model from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    config = ckpt["config"]  # ✅ 이미 GPT2Config 객체
    model = GPT2Model(config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, config

def quantize_model(model: torch.nn.Module) -> torch.nn.Module:
    """모델을 quantize하는 함수"""
    print("⚙️ Applying dynamic quantization...")
    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )


def save_model(model: torch.nn.Module, path: str):
    torch.save(model.state_dict(), path)  # 또는 torch.save(model, path) 로 전체 저장 가능
    




if __name__ == "__main__":
    student_ckpt_path = "saved_models/student.pt"  # ✅ 실제 저장된 경로로 수정
    quant_ckpt_path = "saved_models/student_quant.pt"

    print("📦 Loading student model...")
    model, config = load_student_model(student_ckpt_path)

    print("⚙️ Applying quantization...")
    quant_model = quantize_model(model)

    print("💾 Saving quantized model...")

    save_model(quant_model, quant_ckpt_path)
    #save_model(quant_model, quant_ckpt_path, config=model.config)해야 모델 불러올 수 있음 아니면 학습 끝난 모델에서 config 추출
  

    print(f"✅ Quantized model saved to {quant_ckpt_path}")
    
    # 테스트: 저장된 모델 다시 로드해보기
    print("\n🧪 Testing model loading...")
    test_model, test_config = load_quantized_model(quant_ckpt_path)
    print(f"✅ Successfully loaded quantized model with config:")
    print(f"  - Hidden size: {test_config.hidden_size}")
    print(f"  - Num layers: {test_config.num_hidden_layers}")
    print(f"  - Num attention heads: {test_config.num_attention_heads}")