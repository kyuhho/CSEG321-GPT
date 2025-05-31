import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import GPT2LMHeadModel
from config import GPT2Config
from models.gpt2 import GPT2Model


def load_finetuned_weights(hf_model_name: str = "gavin124/gpt2-finetuned-cnn-summarization-v2"):
    hf_model = GPT2LMHeadModel.from_pretrained(hf_model_name)
    hf_sd = hf_model.transformer.state_dict()

    config = GPT2Config(
        hidden_size=hf_model.config.hidden_size,
        num_hidden_layers=hf_model.config.n_layer,
        num_attention_heads=hf_model.config.n_head,
        intermediate_size=hf_model.config.n_inner or hf_model.config.hidden_size * 4,
        vocab_size=hf_model.config.vocab_size,
        pad_token_id=hf_model.config.pad_token_id,
        max_position_embeddings=hf_model.config.n_positions,
        hidden_dropout_prob=hf_model.config.embd_pdrop,
        layer_norm_eps=hf_model.config.layer_norm_epsilon,
    )

    model = GPT2Model(config)
    print(f"Initialized custom GPT2Model with {config.num_hidden_layers} layers")

    model.word_embedding.weight.data = hf_sd["wte.weight"]
    model.pos_embedding.weight.data = hf_sd["wpe.weight"]

    for i in range(config.num_hidden_layers):
        prefix = f"h.{i}"
        c_attn_w = hf_sd[f"{prefix}.attn.c_attn.weight"]
        c_attn_b = hf_sd[f"{prefix}.attn.c_attn.bias"]
        d = config.hidden_size
        block = model.gpt_layers[i]

        block.self_attention.query.weight.data = c_attn_w[:, :d].T
        block.self_attention.query.bias.data = c_attn_b[:d]
        block.self_attention.key.weight.data = c_attn_w[:, d:2*d].T
        block.self_attention.key.bias.data = c_attn_b[d:2*d]
        block.self_attention.value.weight.data = c_attn_w[:, 2*d:].T
        block.self_attention.value.bias.data = c_attn_b[2*d:]

        block.attention_dense.weight.data = hf_sd[f"{prefix}.attn.c_proj.weight"].T
        block.attention_dense.bias.data = hf_sd[f"{prefix}.attn.c_proj.bias"]

        block.attention_layer_norm.weight.data = hf_sd[f"{prefix}.ln_1.weight"]
        block.attention_layer_norm.bias.data = hf_sd[f"{prefix}.ln_1.bias"]

        block.interm_dense.weight.data = hf_sd[f"{prefix}.mlp.c_fc.weight"].T
        block.interm_dense.bias.data = hf_sd[f"{prefix}.mlp.c_fc.bias"]
        block.out_dense.weight.data = hf_sd[f"{prefix}.mlp.c_proj.weight"].T
        block.out_dense.bias.data = hf_sd[f"{prefix}.mlp.c_proj.bias"]

        block.out_layer_norm.weight.data = hf_sd[f"{prefix}.ln_2.weight"]
        block.out_layer_norm.bias.data = hf_sd[f"{prefix}.ln_2.bias"]

    model.final_layer_norm.weight.data = hf_sd["ln_f.weight"]
    model.final_layer_norm.bias.data = hf_sd["ln_f.bias"]

    print("✅ GPT2 weights successfully loaded into custom model.")
    return model


if __name__ == "__main__":
    model = load_finetuned_weights()
    save_path = "models/gpt2_custom_cnn.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"💾 Saved state_dict to {save_path}")
