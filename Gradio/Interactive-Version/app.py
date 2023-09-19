import argparse
import glob
import json
import os.path

import time
import datetime
from pytz import timezone

import torch
import torch.nn.functional as F

import gradio as gr

from x_transformer import *
import tqdm

from midi_synthesizer import synthesis
import TMIDIX

import matplotlib.pyplot as plt

in_space = os.getenv("SYSTEM") == "spaces"

# =================================================================================================

@torch.no_grad()
def GenerateMIDI(num_tok, idrums, iinstr):
    print('=' * 70)
    print('Req start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    start_time = time.time()

    print('-' * 70)
    print('Req num tok:', num_tok)
    print('Req instr:', iinstr)
    print('Drums:', idrums)
    print('-' * 70)

    if idrums:
        drums = 3074
    else:
        drums = 3073

    instruments_list = ["Piano", "Guitar", "Bass", "Violin", "Cello", "Harp", "Trumpet", "Sax", "Flute", 'Drums',
                        "Choir", "Organ"]
    first_note_instrument_number = instruments_list.index(iinstr)

    start_tokens = [3087, drums, 3075 + first_note_instrument_number]

    print('Selected Improv sequence:')
    print(start_tokens)
    print('-' * 70)

    output_signature = 'Allegro Music Transformer'
    output_file_name = 'Allegro-Music-Transformer-Music-Composition'
    track_name = 'Project Los Angeles'
    list_of_MIDI_patches = [0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0]
    number_of_ticks_per_quarter = 500
    text_encoding = 'ISO-8859-1'

    output_header = [number_of_ticks_per_quarter,
                     [['track_name', 0, bytes(output_signature, text_encoding)]]]

    patch_list = [['patch_change', 0, 0, list_of_MIDI_patches[0]],
                  ['patch_change', 0, 1, list_of_MIDI_patches[1]],
                  ['patch_change', 0, 2, list_of_MIDI_patches[2]],
                  ['patch_change', 0, 3, list_of_MIDI_patches[3]],
                  ['patch_change', 0, 4, list_of_MIDI_patches[4]],
                  ['patch_change', 0, 5, list_of_MIDI_patches[5]],
                  ['patch_change', 0, 6, list_of_MIDI_patches[6]],
                  ['patch_change', 0, 7, list_of_MIDI_patches[7]],
                  ['patch_change', 0, 8, list_of_MIDI_patches[8]],
                  ['patch_change', 0, 9, list_of_MIDI_patches[9]],
                  ['patch_change', 0, 10, list_of_MIDI_patches[10]],
                  ['patch_change', 0, 11, list_of_MIDI_patches[11]],
                  ['patch_change', 0, 12, list_of_MIDI_patches[12]],
                  ['patch_change', 0, 13, list_of_MIDI_patches[13]],
                  ['patch_change', 0, 14, list_of_MIDI_patches[14]],
                  ['patch_change', 0, 15, list_of_MIDI_patches[15]],
                  ['track_name', 0, bytes(track_name, text_encoding)]]

    output = output_header + [patch_list]

    yield output, None, None, [create_msg("visualizer_clear", None)]

    outy = start_tokens
    
    ctime = 0
    dur = 0
    vel = 90
    pitch = 0
    channel = 0

    for i in range(max(1, min(512, num_tok))):
        
        inp = torch.LongTensor([outy]).cpu()

        out = model.module.generate(inp,
                                    1,
                                    temperature=0.9,
                                    return_prime=False,
                                    verbose=False)

        out0 = out[0].tolist()
        outy.extend(out0)
        
        ss1 = out0[0]

        if 0 < ss1 < 256:
            ctime += ss1 * 8

        if 256 <= ss1 < 1280:
            dur = ((ss1 - 256) // 8) * 32
            vel = (((ss1 - 256) % 8) + 1) * 15

        if 1280 <= ss1 < 2816:
            channel = (ss1 - 1280) // 128
            pitch = (ss1 - 1280) % 128
            event = ['note', ctime, dur, channel, pitch, vel]
            output[-1].append(event)

            yield output, None, None, [create_msg("visualizer_append", event), create_msg("progress", [i + 1, num_tok])]

    midi_data = TMIDIX.score2midi(output, text_encoding)

    with open(f"Allegro-Music-Transformer-Music-Composition.mid", 'wb') as f:
        f.write(midi_data)

    audio = synthesis(TMIDIX.score2opus(output), 'SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2')
    
    print('Sample INTs', outy[:16])
    print('-' * 70)
    print('Last generated MIDI event', output[2][-1])
    print('-' * 70)
    print('Req end time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('-' * 70)
    print('Req execution time:', (time.time() - start_time), 'sec')
    
    yield output, "Allegro-Music-Transformer-Music-Composition.mid", (44100, audio), [
        create_msg("visualizer_end", None)]


def cancel_run(mid_seq):
    if mid_seq is None:
        return None, None, None
    text_encoding = 'ISO-8859-1'
    midi_data = TMIDIX.score2midi(mid_seq, text_encoding)

    with open(f"Allegro-Music-Transformer-Music-Composition.mid", 'wb') as f:
        f.write(midi_data)

    audio = synthesis(TMIDIX.score2opus(mid_seq), 'SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2')

    yield "Allegro-Music-Transformer-Music-Composition.mid", (44100, audio), [
        create_msg("visualizer_end", None)]


# =================================================================================================

def load_javascript(dir="javascript"):
    scripts_list = glob.glob(f"{dir}/*.js")
    javascript = ""
    for path in scripts_list:
        with open(path, "r", encoding="utf8") as jsfile:
            javascript += f"\n<!-- {path} --><script>{jsfile.read()}</script>"
    template_response_ori = gr.routes.templates.TemplateResponse

    def template_response(*args, **kwargs):
        res = template_response_ori(*args, **kwargs)
        res.body = res.body.replace(
            b'</head>', f'{javascript}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response


class JSMsgReceiver(gr.HTML):

    def __init__(self, **kwargs):
        super().__init__(elem_id="msg_receiver", visible=False, **kwargs)

    def postprocess(self, y):
        if y:
            y = f"<p>{json.dumps(y)}</p>"
        return super().postprocess(y)

    def get_block_name(self) -> str:
        return "html"


def create_msg(name, data):
    return {"name": name, "data": data}


if __name__ == "__main__":
    
    PDT = timezone('US/Pacific')
    
    print('=' * 70)
    print('App start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('=' * 70)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    parser.add_argument("--port", type=int, default=7860, help="gradio server port")
    opt = parser.parse_args()
    
    print('Loading model...')

    SEQ_LEN = 2048

    # instantiate the model

    model = TransformerWrapper(
        num_tokens=3088,
        max_seq_len=SEQ_LEN,
        attn_layers=Decoder(dim=1024, depth=16, heads=8)
    )

    model = AutoregressiveWrapper(model)

    model = torch.nn.DataParallel(model)

    model.cpu()
    print('=' * 70)

    print('Loading model checkpoint...')

    model.load_state_dict(
        torch.load('Allegro_Music_Transformer_Tiny_Trained_Model_80000_steps_0.9457_loss_0.7443_acc.pth',
                   map_location='cpu'))
    print('=' * 70)

    model.eval()

    print('Done!')
    print('=' * 70)

    load_javascript()
    app = gr.Blocks()
    with app:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Allegro Music Transformer</h1>")
        gr.Markdown(
            "![Visitors](https://api.visitorbadge.io/api/visitors?path=asigalov61.Allegro-Music-Transformer&style=flat)\n\n"
            "Full-attention multi-instrumental music transformer featuring asymmetrical encoding with octo-velocity, and chords counters tokens, optimized for speed and performance\n\n"
            "Check out [Allegro Music Transformer](https://github.com/asigalov61/Allegro-Music-Transformer) on GitHub!\n\n"
            "Special thanks go out to [SkyTNT](https://github.com/SkyTNT/midi-model) for fantastic FluidSynth Synthesizer and MIDI Visualizer code\n\n"
            "[Open In Colab]"
            "(https://colab.research.google.com/github/asigalov61/Allegro-Music-Transformer/blob/main/Allegro_Music_Transformer_Composer.ipynb)"
            " for faster execution and endless generation"
        )
        js_msg = JSMsgReceiver()
        input_drums = gr.Checkbox(label="Add Drums", value=False, info="Add drums to the composition")
        input_instrument = gr.Radio(
            ["Piano", "Guitar", "Bass", "Violin", "Cello", "Harp", "Trumpet", "Sax", "Flute", "Choir", "Organ"],
            value="Piano", label="Lead Instrument Controls", info="Desired lead instrument")
        input_num_tokens = gr.Slider(16, 512, value=256, label="Number of Tokens", info="Number of tokens to generate")
        run_btn = gr.Button("generate", variant="primary")
        interrupt_btn = gr.Button("interrupt")

        output_midi_seq = gr.Variable()
        output_midi_visualizer = gr.HTML(elem_id="midi_visualizer_container")
        output_audio = gr.Audio(label="output audio", format="mp3", elem_id="midi_audio")
        output_midi = gr.File(label="output midi", file_types=[".mid"])
        run_event = run_btn.click(GenerateMIDI, [input_num_tokens, input_drums, input_instrument],
                                  [output_midi_seq, output_midi, output_audio, js_msg])
        interrupt_btn.click(cancel_run, output_midi_seq, [output_midi, output_audio, js_msg],
                            cancels=run_event, queue=False)
        app.queue(concurrency_count=1).launch(server_port=opt.port, share=opt.share, inbrowser=True)