# To-do List

- Ensure better automatic documentation of tensorboard audio & spectrogram logs (automatically save the model name and prompt yourself to document what you changed in the particular run).

- Find a way to listen to the output of your latest model

- Ensure saved models will not overwrite existing models

- Merge `test_model.py` with `load_GAN.py`

- Set up training checkpoints

- Maybe auto-save models if you interrupt training

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

- Fix "Check your callbacks" warning

- Check out other portfolio READMEs and update yours (compare with last year's portfolio project)

## Things to try
- Use ReLU

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

## Done
- In GAN_Monitor, you should be able to simply infer the number of latent dimensions from the model
