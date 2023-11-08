import gradio as gr
from ocr.api.recogniser import Recogniser

def wrapper(input):
    
    recogniser = Recogniser()
    text = recogniser.recognise(input)
    
    return text


def main():
    
    
    with gr.Blocks() as demo:
        with gr.Row():
            input = gr.Image("test.jpg",scale= 2)
            output = gr.Textbox(label = "OCR Text")
        with gr.Row():
            btn = gr.Button("OCR handwritten Text")
            
        btn.click(
            wrapper,
            inputs=[input],
            outputs=[output])
    
    
    demo.launch(
    server_name= "0.0.0.0", # for using on make up
    share=False)
    


if __name__ == '__main__':
    main()