### astro_CIV_model
# Guide
```
# load model...
import sys
sys.path.append('../')
import model as UNet
import torch.nn as nn
model = UNet.UNet1Sigmoid(1,1,32,25)
model = nn.DataParallel(model, device_ids=[1])
model.load_state_dict(torch.load("model.pth"))
# use model...
out = model(torch_array)
```
