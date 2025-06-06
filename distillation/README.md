### `train_distillation.py`

```
python distillation/train_distillation.py --train_samples 5000 --val_samples 500 --batch_size 16 --epochs 5 --lr 1e-4
```

### `inference_distillation.py`

```
python -m distillation.inference_distillation --checkpoint saved_models/distilled_gpt2.pt --max_length 512 --num_to_sample 3 --gen_max_length 100 --seed 1234
```
