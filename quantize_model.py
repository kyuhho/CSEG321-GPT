# quantize_model.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
torch.backends.quantized.engine = 'qnnpack'  # 필수 설정

from models.gpt2 import GPT2Model as StudentGPT2Model
from config import GPT2Config

torch.serialization.add_safe_globals([GPT2Config])

def load_student_model(ckpt_path, device='cpu'):
    """저장된 학생 모델을 로드"""
    print(f"Loading student model from {ckpt_path}")
    
    # 체크포인트 로드
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # config 복원
    config = checkpoint['config']
    
    # 모델 생성
    model = StudentGPT2Model(config)
    
    # state_dict 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def quantize_model(model):
    """모델을 양자화"""
    # 양자화 준비
    model.eval()
    
    # 동적 양자화 적용
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    return quantized_model

def save_quantized_model(model, config, path):
    """양자화된 모델 저장"""
    torch.save({
        'quantized_model': model,  # 양자화된 모델 전체 저장
        'config': config
    }, path)

def load_quantized_model(path, device='cpu'):
    """양자화된 모델 로드"""
    print(f"Loading quantized model from {path}")
    checkpoint = torch.load(path, map_location=device)
    
    quantized_model = checkpoint['quantized_model']
    config = checkpoint['config']
    
    return quantized_model, config

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    student_ckpt_path = "saved_models/student_model.pt"  # 위에서 저장한 경로
    quant_ckpt_path = "saved_models/student_quant.pt"

    print("📦 Loading student model...")
    model = load_student_model(student_ckpt_path, device)
    
    print("⚙️  Applying quantization...")
    quant_model = quantize_model(model)
    
    # config도 함께 저장
    checkpoint = torch.load(student_ckpt_path, map_location='cpu')
    config = checkpoint['config']
    
    print("💾 Saving quantized model...")
    save_quantized_model(quant_model, config, quant_ckpt_path)
    
    print(f"✅ Quantized model saved to {quant_ckpt_path}")