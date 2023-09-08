# -*- coding: utf-8 -*-
"""Allegro_Music_Transformer_Composer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hYR_4sSs2kKpcfcfBgV7-1BTqmZBm52A

# Allegro Music Transformer Composer (ver. 1.0)

***

Powered by tegridy-tools: https://github.com/asigalov61/tegridy-tools

***

WARNING: This complete implementation is a functioning model of the Artificial Intelligence. Please excercise great humility, care, and respect. https://www.nscai.gov/

***

#### Project Los Angeles

#### Tegridy Code 2023

***

# (GPU CHECK)
"""

#@title NVIDIA GPU check
!nvidia-smi

"""# (SETUP ENVIRONMENT)"""

#@title Install dependencies
!git clone --depth 1 https://github.com/asigalov61/Allegro-Music-Transformer
!pip install huggingface_hub
!pip install torch
!pip install einops
!pip install torch-summary
!pip install sklearn
!pip install tqdm
!pip install matplotlib
!apt install fluidsynth #Pip does not work for some reason. Only apt works
!pip install midi2audio

# Commented out IPython magic to ensure Python compatibility.
#@title Import modules

print('=' * 70)
print('Loading core Allegro Music Transformer modules...')

import os
import pickle
import random
import secrets
import statistics
import tqdm

from huggingface_hub import hf_hub_download

print('=' * 70)
print('Loading main Allegro Music Transformer modules...')
import torch

# %cd /content/Allegro-Music-Transformer

import TMIDIX
from x_transformer import TransformerWrapper, Decoder, AutoregressiveWrapper

# %cd /content/
print('=' * 70)
print('Loading aux Allegro Music Transformer modules...')

import matplotlib.pyplot as plt

from torchsummary import summary
from sklearn import metrics

from midi2audio import FluidSynth
from IPython.display import Audio, display

print('=' * 70)
print('Done!')
print('Enjoy! :)')
print('=' * 70)

"""# (LOAD MODEL)"""

#@title Load Allegro Music Transformer Small Model

#@markdown Fast model, 32 layers, 225k MIDIs training corpus

full_path_to_model_checkpoint = "/content/Allegro-Music-Transformer/Models/Small/Allegro_Music_Transformer_Small_Trained_Model_56000_steps_0.9399_loss_0.7374_acc.pth" #@param {type:"string"}
plot_tokens_embeddings = True # @param {type:"boolean"}

print('=' * 70)
print('Loading Allegro Music Transformer Small Pre-Trained Model...')
print('Please wait...')
print('=' * 70)
hf_hub_download(repo_id='asigalov61/Allegro-Music-Transformer',
                filename='Allegro_Music_Transformer_Small_Trained_Model_56000_steps_0.9399_loss_0.7374_acc.pth',
                local_dir='/content/Allegro-Music-Transformer/Models/Small/',
                local_dir_use_symlinks=False)
print('=' * 70)
print('Instantiating model...')

SEQ_LEN = 2048

# instantiate the model

model = TransformerWrapper(
    num_tokens = 3088,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(dim = 1024, depth = 32, heads = 8)
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

# Model stats
print('Model summary...')
summary(model)

# Plot Token Embeddings

if plot_tokens_embeddings:

  tok_emb = model.module.net.token_emb.emb.weight.detach().cpu().tolist()

  cos_sim = metrics.pairwise_distances(
    tok_emb, metric='euclidean'
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

"""# (LOAD SEED MIDI)"""

#@title Load Seed MIDI
select_seed_MIDI = "Allegro-Music-Transformer-Piano-Seed-1" #@param ["Allegro-Music-Transformer-Piano-Seed-1", "Allegro-Music-Transformer-Piano-Seed-2", "Allegro-Music-Transformer-Piano-Seed-3", "Allegro-Music-Transformer-Piano-Seed-4", "Allegro-Music-Transformer-Piano-Seed-5", "Allegro-Music-Transformer-MI-Seed-1", "Allegro-Music-Transformer-MI-Seed-2", "Allegro-Music-Transformer-MI-Seed-3", "Allegro-Music-Transformer-MI-Seed-4", "Allegro-Music-Transformer-MI-Seed-5"]
full_path_to_custom_seed_MIDI = "" #@param {type:"string"}
number_of_prime_tokens = 402 #@param {type:"slider", min:126, max:3000, step:3}
render_MIDI_to_audio = True # @param {type:"boolean"}

if full_path_to_custom_seed_MIDI == '':
  f = '/content/Allegro-Music-Transformer/Seeds/'+select_seed_MIDI+'.mid'

else:
  f = full_path_to_custom_seed_MIDI

print('=' * 70)
print('Allegro Music Transformer Seed MIDI Loader')
print('=' * 70)
print('Loading seed MIDI...')
print('=' * 70)
print('File:', f)
print('=' * 70)

#=======================================================
# START PROCESSING

melody_chords_f = []

# Convering MIDI to ms score with MIDI.py module
score = TMIDIX.midi2ms_score(open(f, 'rb').read())

# INSTRUMENTS CONVERSION CYCLE
events_matrix = []
itrack = 1
patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

patch_map = [
            [0, 1, 2, 3, 4, 5, 6, 7], # Piano
            [24, 25, 26, 27, 28, 29, 30], # Guitar
            [32, 33, 34, 35, 36, 37, 38, 39], # Bass
            [40, 41], # Violin
            [42, 43], # Cello
            [46], # Harp
            [56, 57, 58, 59, 60], # Trumpet
            [64, 65, 66, 67, 68, 69, 70, 71], # Sax
            [72, 73, 74, 75, 76, 77, 78], # Flute
            [-1], # Drums
            [52, 53], # Choir
            [16, 17, 18, 19, 20] # Organ
            ]

while itrack < len(score):
    for event in score[itrack]:
        if event[0] == 'note' or event[0] == 'patch_change':
            events_matrix.append(event)
    itrack += 1

events_matrix.sort(key=lambda x: x[1])

events_matrix1 = []

for event in events_matrix:
        if event[0] == 'patch_change':
            patches[event[2]] = event[3]

        if event[0] == 'note':
            event.extend([patches[event[3]]])
            once = False

            for p in patch_map:
                if event[6] in p and event[3] != 9: # Except the drums
                    event[3] = patch_map.index(p)
                    once = True

            if not once and event[3] != 9: # Except the drums
                event[3] = 15 # All other instruments/patches channel
                event[5] = max(80, event[5])

            if event[3] < 12: # We won't write chans 12-16 for now...
                events_matrix1.append(event)

if len(events_matrix1) > 0:


    #=======================================================
    # PRE-PROCESSING

    # checking number of instruments in a composition
    instruments_list_without_drums = list(set([y[3] for y in events_matrix1 if y[3] != 9]))

    if len(events_matrix1) > 0 and len(instruments_list_without_drums) > 0:

      # recalculating timings
      for e in events_matrix1:
          e[1] = int(e[1] / 8) # Max 2 seconds for start-times
          e[2] = int(e[2] / 32) # Max 4 seconds for durations

      # Sorting by pitch, then by start-time
      events_matrix1.sort(key=lambda x: x[4], reverse=True)
      events_matrix1.sort(key=lambda x: x[1])

      #=======================================================
      # FINAL PRE-PROCESSING

      melody_chords = []

      pe = events_matrix1[0]

      for e in events_matrix1:

          # Cliping all values...
          time = max(0, min(255, e[1]-pe[1]))
          dur = max(1, min(127, e[2]))
          cha = max(0, min(11, e[3]))
          ptc = max(1, min(127, e[4]))

          # Calculating octo-velocity
          vel = max(8, min(127, e[5]))
          velocity = round(vel / 15)-1

          # Writing final note
          melody_chords.append([time, dur, cha, ptc, velocity])

          pe = e

      times = [y[0] for y in melody_chords[12:]]
      avg_time = sum(times) / len(times)

      times_list = list(set(times))

      mode_dur = statistics.mode([y[1] for y in melody_chords if y[2] != 9])

      instruments_list = list(set([y[2] for y in melody_chords]))
      num_instr = len(instruments_list)

      #=======================================================

      # TOTAL DICTIONARY SIZE 3087+1=3088

      #=======================================================
      # MAIN PROCESSING CYCLE
      #=======================================================

      chords_count = 0

      melody_chords_f.extend([2816]) # Zero chords count

      if melody_chords[0][0] == 0:
        melody_chords_f.extend([0]) # Zero time, if present

      for m in melody_chords:

        time = m[0]

        # Chords counter token
        if chords_count % 50 == 0 and chords_count != 0 and time != 0:
          melody_chords_f.extend([2816+min(255, ((chords_count // 50)))])

        if time != 0:
          chords_count += 1

        # WRITING EACH NOTE HERE
        dur_vel = (m[1] * 8) + m[4]
        cha_ptc = (m[2] * 128) + m[3]

        if time != 0:
            melody_chords_f.extend([time, dur_vel+256, cha_ptc+1280])

        else:
            melody_chords_f.extend([dur_vel+256, cha_ptc+1280])

melody_chords_f = melody_chords_f[:number_of_prime_tokens]

#=======================================================

song = melody_chords_f

song_f = []

time = 0
dur = 0
vel = 90
pitch = 0
channel = 0

for ss in song:

  if ss > 0 and ss < 256:

      time += ss * 8

  if ss >= 256 and ss < 1280:

      dur = ((ss-256) // 8) * 32
      vel = (((ss-256) % 8)+1) * 15

  if ss >= 1280 and ss < 2816:
      channel = (ss-1280) // 128
      pitch = (ss-1280) % 128

      song_f.append(['note', time, dur, channel, pitch, vel ])

detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                      output_signature = 'Allegro Music Transformer',
                                                      output_file_name = '/content/Allegro-Music-Transformer-Seed-Composition',
                                                      track_name='Project Los Angeles',
                                                      list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                      number_of_ticks_per_quarter=500)

#=======================================================

print('=' * 70)
print('Composition stats:')
print('Composition has', len(melody_chords_f), 'tokens')
print('=' * 70)

print('Displaying resulting composition...')
print('=' * 70)

fname = '/content/Allegro-Music-Transformer-Seed-Composition'

x = []
y =[]
c = []

colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'white', 'gold', 'silver']

block_lines = [(song_f[-1][1] / 1000)]
block_tokens = [number_of_prime_tokens]

for s in song_f:
  x.append(s[1] / 1000)
  y.append(s[4])
  c.append(colors[s[3]])

if render_MIDI_to_audio:
  FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))
  display(Audio(str(fname + '.wav'), rate=16000))

plt.figure(figsize=(14,5))
ax=plt.axes(title=fname)
ax.set_facecolor('black')

plt.scatter(x,y, c=c)
plt.xlabel("Time")
plt.ylabel("Pitch")
plt.show()

"""# (COMPOSITION LOOP)

## Run the cells below in a loop to generate endless continuation
"""

#@title Standard/Simple Continuation
number_of_tokens_to_generate = 120 # @param {type:"slider", min:33, max:1023, step:3}
number_of_batches_to_generate = 1 #@param {type:"slider", min:1, max:16, step:1}
preview_length_in_tokens = 120 # @param {type:"slider", min:33, max:240, step:3}
number_of_memory_tokens = 2048 # @param {type:"slider", min:402, max:2048, step:2}
temperature = 0.9 #@param {type:"slider", min:0.1, max:1, step:0.1}
render_MIDI_to_audio = True # @param {type:"boolean"}

print('=' * 70)
print('Allegro Music Transformer Standard Model Generator')
print('=' * 70)

preview = melody_chords_f[-preview_length_in_tokens:]

inp = [melody_chords_f[-number_of_memory_tokens:]] * number_of_batches_to_generate

inp = torch.LongTensor(inp).cuda()

out = model.module.generate(inp,
                            number_of_tokens_to_generate,
                            temperature=temperature,
                            return_prime=False,
                            verbose=True)

out0 = out.tolist()

print('=' * 70)
print('Done!')

#======================================================================
print('=' * 70)
print('Rendering results...')

for i in range(number_of_batches_to_generate):

  print('=' * 70)
  print('Batch #', i)
  print('=' * 70)

  out1 = out0[i]

  print('Sample INTs', out1[:12])
  print('=' * 70)

  if len(out) != 0:

      song = preview + out1
      song_f = []
      time = 0
      dur = 0
      vel = 0
      pitch = 0
      channel = 0

      for ss in song:

        if ss > 0 and ss < 256:

            time += ss * 8

        if ss >= 256 and ss < 1280:

            dur = ((ss-256) // 8) * 32
            vel = (((ss-256) % 8)+1) * 15

        if ss >= 1280 and ss < 2816:
            channel = (ss-1280) // 128
            pitch = (ss-1280) % 128

            song_f.append(['note', time, dur, channel, pitch, vel ])

      detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = 'Allegro Music Transformer',
                                                          output_file_name = '/content/Allegro-Music-Transformer-Music-Composition_'+str(i),
                                                          track_name='Project Los Angeles',
                                                          list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                          number_of_ticks_per_quarter=500)
      print('=' * 70)
      print('Displaying resulting composition...')
      print('=' * 70)

      fname = '/content/Allegro-Music-Transformer-Music-Composition_'+str(i)

      x = []
      y =[]
      c = []

      colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'white', 'gold', 'silver']

      for s in song_f:
        x.append(s[1] / 1000)
        y.append(s[4])
        c.append(colors[s[3]])

      if render_MIDI_to_audio:
        FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))
        display(Audio(str(fname + '.wav'), rate=16000))

      plt.figure(figsize=(14,5))
      ax=plt.axes(title=fname)
      ax.set_facecolor('black')

      plt.scatter(x,y, c=c)

      pbl = song_f[(int(preview_length_in_tokens / 3))][1] / 1000

      ax.axvline(x=pbl, c='w')

      plt.xlabel("Time")
      plt.ylabel("Pitch")
      plt.show()

#@title Choose one generated block to add to the composition
block_action = "add_last_generated_block" #@param ["add_last_generated_block", "remove_last_added_block"]
add_block_with_batch_number = 0 #@param {type:"slider", min:0, max:15, step:1}
render_MIDI_to_audio = True # @param {type:"boolean"}

print('=' * 70)

if block_action == 'add_last_generated_block':
  melody_chords_f.extend(out0[min(len(out0)-1, add_block_with_batch_number)])
  print('Block added!')
else:
  if len(melody_chords_f) > number_of_prime_tokens:
    melody_chords_f = melody_chords_f[:max(number_of_prime_tokens, (len(melody_chords_f)-block_tokens[-1]))]
    print('Block removed!')
  else:
    print('Nothing to remove!!!')

print('=' * 70)
print('Composition now has', len(melody_chords_f), 'tokens')


print('=' * 70)
print('Sample INTs', out1[:12])
print('=' * 70)

if len(melody_chords_f) != 0:

    song = melody_chords_f
    song_f = []
    time = 0
    dur = 0
    vel = 0
    pitch = 0
    channel = 0

    for ss in song:

      if ss > 0 and ss < 256:

          time += ss * 8

      if ss >= 256 and ss < 1280:

          dur = ((ss-256) // 8) * 32
          vel = (((ss-256) % 8)+1) * 15

      if ss >= 1280 and ss < 2816:
          channel = (ss-1280) // 128
          pitch = (ss-1280) % 128

          song_f.append(['note', time, dur, channel, pitch, vel ])

    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                        output_signature = 'Allegro Music Transformer',
                                                        output_file_name = '/content/Allegro-Music-Transformer-Music-Composition',
                                                        track_name='Project Los Angeles',
                                                        list_of_MIDI_patches=[0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 53, 19, 0, 0, 0, 0],
                                                        number_of_ticks_per_quarter=500)
    print('=' * 70)
    print('Displaying resulting composition...')
    print('=' * 70)

    fname = '/content/Allegro-Music-Transformer-Music-Composition'

    x = []
    y =[]
    c = []

    colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'white', 'gold', 'silver']

    if block_action == 'add_last_generated_block':
      block_lines.append((song_f[-1][1] / 1000))
      block_tokens.append(len(out0[min(len(out0)-1, add_block_with_batch_number)]))
    else:
      if len(block_lines) > 1:
        block_lines.pop()
        block_tokens.pop()

    for s in song_f:
      x.append(s[1] / 1000)
      y.append(s[4])
      c.append(colors[s[3]])

    if render_MIDI_to_audio:
      FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))
      display(Audio(str(fname + '.wav'), rate=16000))

    plt.figure(figsize=(14,5))
    ax=plt.axes(title=fname)
    ax.set_facecolor('black')

    plt.scatter(x,y, c=c)

    for bl in block_lines:
      ax.axvline(x=bl, c='w')

    plt.xlabel("Time")
    plt.ylabel("Pitch")
    plt.show()

"""# Congrats! You did it! :)"""