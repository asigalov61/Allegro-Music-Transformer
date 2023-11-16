print('=' * 70)
print('Loading core Allegro Music Transformer modules...')

import os
import pickle
import random
import secrets
import statistics
from time import time
import tqdm

print('=' * 70)
print('Loading main Allegro Music Transformer modules...')
import torch

import TMIDIX
from x_transformer import *

print('=' * 70)
print('Loading aux Allegro Music Transformer modules...')

import matplotlib.pyplot as plt

from torchsummary import summary
from sklearn import metrics

from midi2audio import FluidSynth
from IPython.display import Audio, display

from huggingface_hub import hf_hub_download

from google.colab import files

print('=' * 70)
print('Done!')

# Load configuration from a JSON file
def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

# Load and set up the model based on the configuration
def setup_model(model_config):
    # Load the model configuration
    full_path_to_model_checkpoint = model_config['full_path_to_model_checkpoint']
    
    # Load the model precision
    model_precision = model_config['model_precision']
    
    # Chose if the plot tokens embeddings should be plotted
    plot_tokens_embeddings = model_config['plot_tokens_embeddings']
    
    print('=' * 70)
    print('Loading Allegro Music Transformer Tiny Pre-Trained Model...')
    print('Please wait...')
    print('=' * 70)
    
    if os.path.isfile(full_path_to_model_checkpoint):
        print('Model already exists...')
    else:
        hf_hub_download(repo_id='asigalov61/Allegro-Music-Transformer',
                        filename='Allegro_Music_Transformer_Tiny_Trained_Model_80000_steps_0.9457_loss_0.7443_acc.pth',
                        local_dir='/content/Allegro-Music-Transformer/Models/Tiny/',
                        local_dir_use_symlinks=False)
    print('=' * 70)
    print('Instantiating model...')
    
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda'

    if model_precision == 'bfloat16' and torch.cuda.is_bf16_supported():
        dtype = 'bfloat16'
    else:
        dtype = 'float16'

    if model_precision == 'float16':
        dtype = 'float16'

    if model_precision == 'float32':
        dtype = 'float32'

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        
    # instantiate the model

    model = TransformerWrapper(
        num_tokens = model_config['num_tokens'],
        max_seq_len = model_config['seq_len'],
        attn_layers = Decoder(dim = 1024, depth = 16, heads = 8, attn_flash=True)
    )

    model = AutoregressiveWrapper(model)

    model = torch.nn.DataParallel(model)

    model.cuda()
    print('=' * 70)

    print('Loading model checkpoint...')

    model.load_state_dict(torch.load(full_path_to_model_checkpoint))
    print('=' * 70)

    model.eval()

    print('Done!')
    print('=' * 70)

    print('Model will use', dtype, 'precision...')
    print('=' * 70)

    return model

def model_stats(model_config, model):
    print('Model summary...')
    summary(model)

    # Plot Token Embeddings

    if model_config['plot_tokens_embeddings']:
        tok_emb = model.module.net.token_emb.emb.weight.detach().cpu().tolist()

        cos_sim = metrics.pairwise_distances(
            tok_emb, metric='cosine'
        )
        plt.figure(figsize=(7, 7))
        plt.imshow(cos_sim, cmap="inferno", interpolation="nearest")
        im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
        plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
        plt.xlabel("Position")
        plt.ylabel("Position")
        plt.tight_layout()
        plt.plot()
        plt.savefig("/content/Allegro-Music-Transformer-Small-Tokens-Embeddings-Plot.png", bbox_inches="tight")


# Seed MIDI processing
def process_seed_midi(seed_midi_config):
    # ... Seed MIDI processing code ...
    pass

# Continuation generation
def generate_continuation(continuation_config, model):
    # ... Continuation generation code ...
    pass

# Inpainting process
def perform_inpainting(inpainting_config, model):
    # ... Inpainting code ...
    pass

if __name__ == "__main__":
    # Load the configuration file
    config = load_config('config.json')

    # Set up the model
    model = setup_model(config['model'])

    # Process the seed MIDI
    seed_midi_data = process_seed_midi(config['seed_midi'])

    # Generate music based on the general configuration
    generated_music = generate_music(config['general'], model)

    # Perform continuation if specified
    if config['continuation'].get('try_to_generate_outro'):
        continuation_music = generate_continuation(config['continuation'], model)

    # Perform inpainting if any instrument is set to True
    if any(value for key, value in config['inpainting'].items() if key != "number_of_memory_tokens" and key != "number_of_samples_per_inpainted_note"):
        inpainted_music = perform_inpainting(config['inpainting'], model)
