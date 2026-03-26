import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# 여기를 본인 이미지 경로로 바꾸세요
image_names = [
    "examples/kitchen/images/00.png",
    "examples/kitchen/images/01.png",
    "examples/kitchen/images/02.png",
]

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)

print(type(predictions))
print("inference done")
