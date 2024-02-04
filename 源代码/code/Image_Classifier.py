import torch
import timm,os
from PIL import Image
import numpy as np
import json
from Myname_Translator import translator_func
model = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True,num_classes=289)
device = "cuda" if torch.cuda.is_available() else "cpu"
_exp_name = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"

model_best = model.to(device)
model_best.load_state_dict(torch.load(f"../model/{_exp_name}_best.ckpt"))
model_best.eval()

data_config = timm.data.resolve_model_data_config(model)
test_tfm = timm.data.create_transform(**data_config, is_training=False)

class ImageProcessor:
  def __init__(self,image_path,data=None):
    try:
      print("Image input:{}".format(image_path))
      if isinstance(image_path, str):
        self.image = Image.open(image_path).convert('RGB')
      elif isinstance(image_path, np.ndarray):
        self.image = Image.fromarray(image_path).convert('RGB')
      print("Image loaded successfully.")
      self.data = data
    except Exception as e:
      print("Error loading:", str(e))

  def load_data(self):
    file_path = 'D:\毕业设计\源代码\data\picture\label.json'
    # 打开文件并读取内容
    try:
      with open(file_path, 'r', encoding='utf-8') as file:
        # 解析JSON数据
        idtoname = json.load(file)
      print('load data 成功')
      # data现在是一个包含键值对的Python字典
      self.data=idtoname
    except Exception as e:
      print(e)

  def func(self):
    self.load_data()
    img = self.image
    id2name = self.data
    try:
      output = model_best(test_tfm(img).unsqueeze(0).cuda())
      pred = np.argmax(output.cpu().data.numpy(), axis=1)
      pred = pred[0]
      return id2name[pred]['名称']
    except Exception as e:
      print('func:',e)

def Image_Classifier(image_path):
  myprocessor = ImageProcessor(image_path=image_path)
  label = myprocessor.func()
  answer = translator_func(label)[1]
  return label,answer

Image_Classifier("../data/picture/一串红/0_0.jpg")
