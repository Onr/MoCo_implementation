from model import LitMoCo
import torch
from pytorch_lightning import Trainer

# net = LitMoCo()
x = torch.randn(1, 1, 28, 28)

model = LitMoCo()
trainer = Trainer(max_steps=10)
trainer.fit(model)

out = model(x)
print(out.shape)