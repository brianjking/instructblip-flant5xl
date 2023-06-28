import gradio as gr 
from lavis.models import load_model_and_preprocess
import torch

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model_name = "blip2_t5_instruct"
model_type = "flant5xl"
model, vis_processors, _ = load_model_and_preprocess(
    name=model_name,
    model_type=model_type,
    is_eval=True,
    device=device
)

def infer(image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, decoding_method):
        use_nucleus_sampling = decoding_method == "Nucleus sampling"
        image = vis_processors["eval"](image).unsqueeze(0).to(device)

        samples = {
            "image": image,
            "prompt": prompt,
        }

        output = model.generate(
            samples,
            length_penalty=float(len_penalty),
            repetition_penalty=float(repetition_penalty),
            num_beams=beam_size,
            max_length=max_len,
            min_length=min_len,
            top_p=top_p,
            use_nucleus_sampling=use_nucleus_sampling
        )

        return output[0]
    
theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)
css = ".generating {visibility: hidden}"

examples = [
["banff.jpg", "Can you tell me about this image in detail", 1, 200, 5, 1, 3, 0.9, "Beam search"]
]
with gr.Blocks(theme=theme, analytics_enabled=False,css=css) as demo:
    gr.Markdown("## InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning")
    gr.Markdown(
            """
            Unofficial demo for InstructBLIP. InstructBLIP is a new vision-language instruction-tuning framework by Salesforce that uses BLIP-2 models, achieving state-of-the-art zero-shot generalization performance on a wide range of vision-language tasks.
            The demo is based on the official <a href="https://github.com/salesforce/LAVIS/tree/main/projects/instructblip" style="text-decoration: underline;" target="_blank"> Github </a> implementation
            """
        )
    gr.HTML("<p>You can duplicate this Space to run it privately without a queue for shorter queue times  : <a style='display:inline-block' href='https://huggingface.co/spaces/RamAnanth1/InstructBLIP?duplicate=true'><img src='https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14' alt='Duplicate Space'></a> </p>")

    with gr.Row():
        with gr.Column(scale=3):
            image_input = gr.Image(type="pil")
            prompt_textbox = gr.Textbox(label="Prompt:", placeholder="prompt", lines=2)
            output = gr.Textbox(label="Output")
            submit = gr.Button("Run", variant="primary")

        with gr.Column(scale=1):
            min_len = gr.Slider(
                minimum=1,
                maximum=50,
                value=1,
                step=1,
                interactive=True,
                label="Min Length",
            )
        
            max_len = gr.Slider(
                minimum=10,
                maximum=500,
                value=250,
                step=5,
                interactive=True,
                label="Max Length",
            )
        
            sampling = gr.Radio(
                choices=["Beam search", "Nucleus sampling"],
                value="Beam search",
                label="Text Decoding Method",
                interactive=True,
            )
        
            top_p = gr.Slider(
                minimum=0.5,
                maximum=1.0,
                value=0.9,
                step=0.1,
                interactive=True,
                label="Top p",
            )
        
            beam_size = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                interactive=True,
                label="Beam Size",
            )
        
            len_penalty = gr.Slider(
                minimum=-1,
                maximum=2,
                value=1,
                step=0.2,
                interactive=True,
                label="Length Penalty",
            )
        
            repetition_penalty = gr.Slider(
                minimum=-1,
                maximum=3,
                value=1,
                step=0.2,
                interactive=True,
                label="Repetition Penalty",
            )
    gr.Examples(
                    examples=examples,
                    inputs=[image_input, prompt_textbox, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, sampling],
                    cache_examples=False,
                    fn=infer,
                    outputs=[output],
                )
    
    submit.click(infer, inputs=[image_input, prompt_textbox, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, sampling], outputs=[output])

demo.queue(concurrency_count=16).launch(debug=True)
