import torch
torch.backends.quantized.engine = 'qnnpack'  # ✅ 필수 추가!

from models.gpt2 import GPT2Model
from config import GPT2Config

# 1. config 정의
config = GPT2Config()

# 2. 학습 안 된 GPT2Model 생성 (무작위 weight)
model = GPT2Model(config)
model.eval()

# 3. 양자화 적용
quant_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

# 4. 임의 입력 생성 (batch_size=1, seq_len=8)
dummy_input = torch.randint(0, config.vocab_size, (1, 8))
dummy_mask = torch.ones_like(dummy_input)

# 5. 추론 전후 출력 비교
with torch.no_grad():
    out_fp32 = model(dummy_input, dummy_mask)
    out_int8 = quant_model(dummy_input, dummy_mask)

print("✅ 양자화 테스트 완료")
print("🔍 FP32 마지막 토큰 hidden:", out_fp32["last_token"][0][:5])
print("🔍 INT8 마지막 토큰 hidden:", out_int8["last_token"][0][:5])