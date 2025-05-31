### `train_distillation.py`

```
python -m distillation.train_distillation --batch_size 8 --epochs 3 --lr 5e-5 --temperature 2.0 --max_length 512 --debug
```

### `inference_distillation.py`
```
python -m distillation.inference_distillation --checkpoint saved_models/distilled_gpt2.pt --max_length 512 --num_to_sample 3 --gen_max_length 100 --seed 1234
```