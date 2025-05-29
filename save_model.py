# # save_student.py
# import torch
# from models.gpt2 import GPT2Model
# from config import GPT2Config

# config = GPT2Config(
#     hidden_size=768,
#     num_hidden_layers=6,
#     num_attention_heads=12,
#     intermediate_size=3072
# )
# model = GPT2Model(config)
# save_student.py
torch.save({
    "state_dict": model.state_dict(),
    "config": config.__dict__,
}, "student.pt")
# 위 코드만 distillation 후 실행시켜주면 잘 저장됩니다.