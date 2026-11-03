# Residual connections

## Train

From this directory:

```bash
uv run residual-connections --epochs 15 --batch-size 64 --augment

# Python module mode (in the project environment)
python -m residual_connections --epochs 15 --batch-size 64 --augment
```

Use `--help` for all options; omit `--augment` to disable data augmentation.

## Import and load the model

```python
import torch
from residual_connections import ResNet

checkpoint = torch.load("resnet.pt", map_location="cpu", weights_only=True)
model = ResNet(n=checkpoint["config"]["blocks_per_stage"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```
