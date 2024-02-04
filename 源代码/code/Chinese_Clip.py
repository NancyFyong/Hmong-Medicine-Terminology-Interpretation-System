import numpy as np
from PIL import Image
import requests
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained(
    '../model/models--OFA-Sys--chinese-clip-vit-base-patch16/snapshots/36e679e65c2a2fead755ae21162091293ad37834')
processor = AutoProcessor.from_pretrained(
    '../model/models--OFA-Sys--chinese-clip-vit-base-patch16/snapshots/36e679e65c2a2fead755ae21162091293ad37834')
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

class TextImageProcessor:
    def __init__(self,text,image_path):
      try:
        print("Image input:{}".format(image_path))
        if isinstance(image_path,str):
          self.image = Image.open(image_path)
        elif isinstance(image_path,np.ndarray):
          self.image = image_path
        print("Image loaded successfully.")
        print("text input:{}".format(text))
        self.text = re.split(r'[,，\s]+', text)
        print("Text loaded successfully.")
        print(self.text)
      except Exception as e:
        print("Error loading:", str(e))

    def compute_image_features(self):
        # compute image feature
      try:
        image = self.image
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize
        self.image_features = image_features
      except Exception as e:
        print("Error loading:", str(e))

    def compute_text_features(self):
        texts = self.text
        inputs = processor(text=texts, padding=True, return_tensors="pt").to(device)
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize
        self.text_features = text_features

    def compute_similar_score(self):
        inputs = processor(text=self.text, images=self.image, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # probs: [[1.2686e-03, 5.4499e-02, 6.7968e-04, 9.4355e-01]]
        print(probs)
        return torch.argmax(probs)

# %%
import re

path="D:\毕业设计\源代码\data\img_2.png"
text= "一枝黄花 牛筋草 车前草 水葫芦 万年青 八月瓜,药材"
def process_image_and_text(path,text):
    my_processor = TextImageProcessor(image_path=path, text=text)
    text=my_processor.text
    my_processor.compute_image_features()
    my_processor.compute_text_features()
    number = my_processor.compute_similar_score()
    print(text)
    print(number)
    print(text[number])
    return text[number]
