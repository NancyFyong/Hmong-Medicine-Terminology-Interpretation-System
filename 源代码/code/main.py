import gradio as gr
from Chinese_Clip import process_image_and_text
import numpy as np
from Myname_Translator import translator_func
from Image_Classifier import Image_Classifier

with gr.Blocks() as demo:
  gr.Markdown("""
    <center>
    <font size=16>
    苗医药术语解释系统
    </font>
    </center>
    """)
  with gr.Tab("基于文本的苗医药术语解释"):
    with gr.Row():
      with gr.Column(scale=1.5):
        From_input = gr.Textbox(label="请输入苗文或中文术语名称")
        translate_button = gr.Button("确定")
      with gr.Column():
        To_image = gr.Image(label="苗医药植物图片")
        To_text = gr.Textbox(label="解释结果")
  with gr.Tab("基于图像的苗医药术语解释"):
    with gr.Row():
      image = gr.Image(label="请输入图像")
      with gr.Column():
        label = gr.Textbox(label="药物名称")
        answer = gr.Textbox(label="解释结果")
      # 术语解释
    submit_button = gr.Button('提交')
  with gr.Tab("基于图片"):
    with gr.Row():
      with gr.Column():
        image_input = gr.Image(label="图像输入")
        label_input = gr.Textbox(label="可能类别")
      with gr.Column():
        text_output = gr.Textbox(label="药物名称")
    with gr.Column():
      text_button = gr.Button("分类")
  gr.Markdown("## 术语解释例子")
  gr.Examples(
    examples=["一点红","一串红"],
    inputs=From_input,
    outputs=[To_image,To_text],
    fn=translator_func,
    cache_examples=True,
  )
  with gr.Accordion("作者信息"):
    gr.Markdown("""
    <center>
    <font size=12>
    学生：冯志勇 指导老师：莫礼平
    </font>
    </center>""")
  text_button.click(process_image_and_text, inputs=[image_input,label_input], outputs=text_output)
  translate_button.click(translator_func, inputs=From_input, outputs=[To_image,To_text])
  submit_button.click(Image_Classifier,inputs=image,outputs=[label,answer])
if __name__ == "__main__":
    demo.launch(share=True)
