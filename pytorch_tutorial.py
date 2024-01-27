# %%
import torch
import ssl
from PIL import Image
from torchvision import transforms

ssl._create_default_https_context = ssl._create_unverified_context

# model = torch.hub.load("hustvl/yolop", "yolop", pretrained=True)
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")

# # %%
# # with open("blackcat.jpg") as f:
# img = Image.open("blackcat.jpg")
# img.show()

# t = transforms.ToTensor()
# features = t(img)

# # %%
# # img = torch.randn(1, 3, 640, 640)
# det_out, da_seg_out, ll_seg_out = model(features)

# # %%
# print(det_out)
# print(det_out[0].shape)
# # %%
