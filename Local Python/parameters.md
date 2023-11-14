# Allegro Music Transformer User Parameters

This document provides a detailed description of the user-defined parameters for the Allegro Music Transformer script. Each parameter is used to control different aspects of the music generation process.

## General Parameters

### `first_note_instrument`
- **Description**: Selects the instrument for the first note of the generated piece.
- **Possible Values**: "Piano", "Guitar", "Bass", "Violin", "Cello", "Harp", "Trumpet", "Sax", "Flute", "Choir", "Organ"
- **Default Value**: "Piano"

### `add_drums`
- **Description**: Determines whether drums should be added to the composition.
- **Possible Values**: `True` or `False`
- **Default Value**: `False`

### `number_of_tokens_tp_generate`
- **Description**: Specifies the total number of tokens (musical events) the model should generate.
- **Possible Values**: Any integer from 30 to 2048
- **Default Value**: 300

### `number_of_batches_to_generate`
- **Description**: The number of separate compositions to generate in a single execution.
- **Possible Values**: Any integer from 1 to 16
- **Default Value**: 4

### `temperature`
- **Description**: Controls the stochastic nature of the generation process.
- **Possible Values**: A float between 0.1 and 1.0
- **Default Value**: 0.9

### `render_MIDI_to_audio`
- **Description**: If enabled, converts the generated MIDI sequence into an audio file.
- **Possible Values**: `True` or `False`
- **Default Value**: `True`

## Seed MIDI Parameters

### `select_seed_MIDI`
- **Description**: Chooses a seed MIDI file to use as a starting point for generation.
- **Possible Values**: "Upload your own custom MIDI" or a list of predefined MIDI filenames
- **Default Value**: "Upload your own custom MIDI"

## Continuation Parameters

### `try_to_generate_outro`
- **Description**: Instructs the model to try and generate an outro for the piece.
- **Possible Values**: `True` or `False`
- **Default Value**: `False`

### `number_of_prime_tokens`
- **Description**: Number of tokens from the seed MIDI to use as a primer for the generation.
- **Possible Values**: Any integer from 3 to 2046
- **Default Value**: 300

### `number_of_tokens_to_generate`
- **Description**: The number of new tokens to generate for a continuation piece.
- **Possible Values**: Any integer from 30 to 2046
- **Default Value**: 300

### `include_prime_tokens_in_generated_output`
- **Description**: Decides if the seed tokens should be included in the final output.
- **Possible Values**: `True` or `False`
- **Default Value**: `True`

### `allow_model_to_stop_generation_if_needed`
- **Description**: Allows the model to stop generation early if a musical conclusion is reached.
- **Possible Values**: `True` or `False`
- **Default Value**: `False`

## Inpainting Parameters

### `inpaint_instrument`
- **Description**: Specifies which instruments should be subject to inpainting.
- **Possible Values**: A list of booleans corresponding to each instrument.
- **Default Value**: Varies based on which instruments are set to `True`.

### `number_of_memory_tokens`
- **Description**: The number of previous tokens the model uses as context for generation.
- **Possible Values**: Any integer from 8 to 2044
- **Default Value**: 2044

### `number_of_samples_per_inpainted_note`
- **Description**: The number of variations to generate for each inpainted note.
- **Possible Values**: Any integer from 1 to 16
- **Default Value**: 1

---

Please note that the default values are subject to change based on the script implementation and user preferences.
