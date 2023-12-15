
train reference
```bash
accelerate launch train/train_double_unet.py --config config/train/reference_net.yaml
```

train temporal

```bash
accelerate launch train/train_temporal.py --config config/train/rtemporal.yaml
```

validata reference
```
python inference/validate_w_control.py --config config/inference/reference_net.yaml
```

validate video
```
python inference/validate_video.py --config config/inference/temporal.py
```