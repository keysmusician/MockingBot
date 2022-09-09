# To-do List

- Find a way to listen to the output of your latest model

- Ensure better automatic documentation of tensorboard audio & spectrogram logs (automatically save the model name and prompt yourself to document what you changed in the particular run).

- Merge `test_model.py` with `load_GAN.py`

- Set up training checkpoints

- Find more datasets. 

    I need data from a wider range of sources. I currently have about 900 kick drum
    samples, but they're not in a format TensorFlow can open. I want datasets of:
    - Cricket chirps
    - A bird call (how appropriate)
    - Cat meows

    And maybe:
    - Cicada scratches
    - Cow moos
    - Frog croaks
    - Any short and simple animal call, etc.

    I think three sources should suffice. See make_datasets.py

- Make it just as easy to switch to the mock sine dataset as it is to switch to WAV datasets

- Check out other portfolio READMEs and update yours (compare with last year's portfolio project)

- Add a `simulation_count` parameter to `test_generator` to generate and plot more than one example at a time.

- Fix "Check your callbacks" warning

- Fix "WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually." for generator and discriminator models

- Implement a forward pass for GAN.save() to work?

## Things to try
- Increase dataset size

- Use leaky ReLU in discriminator

- Ensure discriminator is architecturally adequate

- Go back to a deconvolution architecture for the generator

- Use ReLU in the generator

- Go back to that LSTM-Dense model but try it on the Meows dataset

- *Don't* normalize the spectrogram (to try to combat the noise floor)

- Decrease layer width?

- Higher learning rate &/ Keep learning rate high for longer

- Parameters from Macaulay Library:

    - Window length: 512 samples

    - STFT hop size: 128 samples

    - Mel scaling: On

    - Image size: [128, 512]

- Augment data with slight pitch shifting
