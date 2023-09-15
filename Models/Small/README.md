# Allegro Music Transformer Small Pre-Trained Model

***

## Model was trained on 225959 MIDIs from LAKH+MMD+GiantMIDI MIDI datasets

***

## Training dataset statistics by instuments notes:

1) Piano: 204336578 notes
2) Guitar: 243333429 notes
3) Bass: 86898124 notes
4) Violin: 5907486 notes
5) Cello: 2946167 notes
6) Harp: 2932013 notes
7) Trumpet: 15137967 notes
8) Sax: 19386453 notes
9) Flute: 14300373 notes
10) Drums: 266442335 notes
11) Choir: 11104911 notes
12) Organ: 15219844 notes

***

## Notes on the results

### 1) Adding chords counters tokens did not improve the result much
### 2) Asymmetrical encoding also showed inferior results (higher loss/lower acc) compared to symmetrical encoding
### 3) The model plays and works exceptionally well :) Probably because of the very same chords counter tokens and sane size. Or maybe it is just the matter of probabilities (read - luck) ?!?

***

### Project Los Angeles
### Tegridy Code 2023
