# quantize_model_tensorrt.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
from models.gpt2 import GPT2Model
from config import GPT2Config
import numpy as np
from tqdm import tqdm

# TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    print("⚠️ TensorRT not available. Installing torch-tensorrt instead...")
    try:
        import torch_tensorrt
        TRT_AVAILABLE = True
    except ImportError:
        print("❌ Neither TensorRT nor torch-tensorrt available. Using PyTorch native optimization.")
        TRT_AVAILABLE = False

torch.serialization.add_safe_globals([GPT2Config])

class CalibrationDataset(Dataset):
    """TensorRT INT8 quantization을 위한 calibration dataset"""
    def __init__(self, tokenizer, max_samples=100, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # CNN/DailyMail validation 데이터의 일부를 사용
        print("📊 Loading calibration dataset...")
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")["validation"]
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.samples = []
        for item in tqdm(dataset, desc="Preparing calibration data"):
            text = f"Article: {item['article'][:800]}\nSummary:"
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.samples.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def load_student_model(checkpoint_path: str, device) -> tuple:
    """Student 모델을 로드하는 함수"""
    print(f"📦 Loading student model from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt["config"]
    model = GPT2Model(config)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, config

def quantize_model_torch_tensorrt(model, tokenizer, device, calibration_samples=50):
    """torch-tensorrt를 사용한 quantization (INT8)"""
    print("⚙️ Applying TensorRT quantization...")
    
    if not TRT_AVAILABLE:
        print("❌ TensorRT not available. Falling back to native PyTorch optimization.")
        return optimize_model_native(model)
    
    try:
        import torch_tensorrt
        
        # Calibration dataset 준비
        calib_dataset = CalibrationDataset(tokenizer, max_samples=calibration_samples)
        calib_dataloader = DataLoader(calib_dataset, batch_size=1, shuffle=False)
        
        # TensorRT 설정
        compile_spec = {
            "inputs": [
                torch_tensorrt.Input(
                    shape=[1, 512],  # [batch_size, sequence_length]
                    dtype=torch.long,
                    name="input_ids"
                ),
                torch_tensorrt.Input(
                    shape=[1, 512],  # [batch_size, sequence_length]
                    dtype=torch.long,
                    name="attention_mask"
                )
            ],
            "enabled_precisions": {torch.int8, torch.float},  # INT8 quantization
            "calibrator": torch_tensorrt.ptq.DataLoaderCalibrator(
                calib_dataloader,
                cache_file=f"./tensorrt_cache_{model.__class__.__name__}.cache",
                use_cache=True,
                algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            ),
            "truncate_long_and_double": True,
            "device": {
                "device_type": torch_tensorrt.DeviceType.GPU,
                "gpu_id": 0,
            }
        }
        
        # TensorRT로 컴파일
        print("🔧 Compiling model with TensorRT...")
        trt_model = torch_tensorrt.compile(model, **compile_spec)
        
        print("✅ TensorRT quantization completed!")
        return trt_model
        
    except Exception as e:
        print(f"❌ TensorRT quantization failed: {e}")
        print("🔄 Falling back to native PyTorch optimization...")
        return optimize_model_native(model)

def optimize_model_native(model):
    """Native PyTorch optimization (fallback)"""
    print("⚙️ Applying native PyTorch optimization...")
    
    # JIT compilation
    model = torch.jit.script(model)
    
    # Optimization passes
    torch.jit.optimize_for_inference(model)
    
    print("✅ Native PyTorch optimization completed!")
    return model

def quantize_model_fake_quant(model, device):
    """Fake quantization for GPU compatibility"""
    print("⚙️ Applying fake quantization for GPU...")
    
    # Fake quantization을 사용하여 INT8 simulation
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 모델을 quantization-aware training 모드로 변경
    torch.quantization.prepare_qat(model, inplace=True)
    
    # Fake quantization 적용
    model.apply(torch.quantization.disable_observer)
    model.apply(torch.quantization.enable_fake_quant)
    
    model = model.to(device)
    model.eval()
    
    print("✅ Fake quantization completed!")
    return model

def save_optimized_model(model, config, path: str, optimization_type: str):
    """최적화된 모델과 config를 함께 저장하는 함수"""
    print(f"💾 Saving {optimization_type} model to {path}")
    
    save_dict = {
        'model': model,
        'config': config,
        'model_type': optimization_type,
        'optimization_info': {
            'type': optimization_type,
            'device_compatible': 'cuda' if optimization_type in ['tensorrt', 'fake_quant'] else 'cpu'
        }
    }
    
    torch.save(save_dict, path, pickle_protocol=4)
    print(f"✅ {optimization_type} model saved successfully!")

def load_optimized_model(checkpoint_path: str, device):
    """최적화된 모델을 로드하는 함수"""
    print(f"📦 Loading optimized model from {checkpoint_path}")
    
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        model = ckpt['model']
        config = ckpt['config']
        optimization_type = ckpt.get('model_type', 'unknown')
        
        model = model.to(device)
        model.eval()
        
        print(f"✅ Loaded {optimization_type} model with config:")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Num layers: {config.num_hidden_layers}")
        print(f"  - Num attention heads: {config.num_attention_heads}")
        print(f"  - Optimization type: {optimization_type}")
        
        return model, config, optimization_type
        
    except Exception as e:
        print(f"❌ Error loading optimized model: {e}")
        raise e

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    student_ckpt_path = "saved_models/student.pt"
    tensorrt_ckpt_path = "saved_models/student_tensorrt.pt"
    
    try:
        # 1. Student 모델 로드
        print("\n📦 Loading student model...")
        model, config = load_student_model(student_ckpt_path, device)
        
        # 2. Tokenizer 로드
        tokenizer = GPT2Tokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 3. 최적화 방법 선택
        optimization_method = "tensorrt"  # 또는 "fake_quant"
        
        if optimization_method == "tensorrt" and device.type == "cuda":
            print("\n🔧 Attempting TensorRT quantization...")
            optimized_model = quantize_model_torch_tensorrt(model, tokenizer, device)
            optimization_type = "tensorrt"
        else:
            print("\n🔧 Using fake quantization...")
            optimized_model = quantize_model_fake_quant(model, device)
            optimization_type = "fake_quant"
        
        # 4. 최적화된 모델 저장
        print("\n💾 Saving optimized model...")
        save_optimized_model(optimized_model, config, tensorrt_ckpt_path, optimization_type)
        
        # 5. 테스트: 저장된 모델 다시 로드해보기
        print("\n🧪 Testing model loading...")
        test_model, test_config, test_type = load_optimized_model(tensorrt_ckpt_path, device)
        print(f"✅ Successfully loaded {test_type} model!")
        
        print(f"\n🎉 Optimization completed!")
        print(f"📍 Optimized model saved to: {tensorrt_ckpt_path}")
        print(f"🔧 Optimization type: {optimization_type}")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 