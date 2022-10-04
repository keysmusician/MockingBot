<div align="center">
    <img src="https://user-images.githubusercontent.com/74752740/190259121-0f200bc3-d668-4e52-bbdf-43888bdc4817.jpg" alt="Pixelated bird">
</div>

# MockingBot
MockingBot is a generative sound effect model which learns to mimic the sound of its training data. Its goal is generate original audio that could convince a human it was not machine-generated.

MockingBot was inspired by what I consider a deficiency in the game *No Man's Sky.* For all its procedurally generated magnificence, the quality of the fauna sounds is not commensurate with the rest of the environment. MockingBot is my attempt to remedy this inadequacy through machine learning.

## Usage
### In Google Colab
Run [the Google Colab demo](https://colab.research.google.com/drive/1O0A5DRFeGcS6ns4gEVVU-dno8D2LkEQ6?usp=sharing), `MockingBot_Demo.ipynb` (Google account required) to display and generate an audio simulation in your browser.

Simply click `Runtime` > `Run All`. The notebook will download the model, import the required libraries, and generate an audio sample which will be displayed in an audio player along with a spectrogram and waveform graph.

### On Your Local Machine
Alternatively, you can download and run the demo [`MockingBot_Demo.ipynb`](https://github.com/keysmusician/MockingBot/blob/main/MockingBot_Demo.ipynb) locally if you have Jupyter and the project dependencies listed under [Technologies](#technologies) installed. The demo model should download automatically, but you can also download it manually [here](https://drive.google.com/drive/folders/1Xe09PWARV-q_iMpH0jnRMhW2vS39OYtb?usp=sharing). Note that if you download the model manually, you'll have to modify the code to load the model from wherever you place it on your local machine.

## Architecture
I tried many architectures (see [Process](#process)). The best one I tried was a GAN with a dense neural network as the generator and a convolutional neural network as the discriminator.

MockingBot works by mapping a series of numbers (a random latent vector) to a spectrogram which should convincingly pass for the spectrogram of its training target (e.g, a cat meow). The spectrogram can then be inverse-Fourier-transformed into an audio signal. The generation process should be fast enough to be suitable for use in real-time situations such as in a video game.

## Input Pipeline
My data preprocessing was relatively simple. As others* have done when training audio models, I trained over spectrograms created from the real component of the short-time fourier transform (STFT) of the training data (which were WAV audio files) rather than directly over the time-domain signals. This has the advantage of being phase-invariant, extracting the frequency components of the audio, as well as enabling convolutional architectures (since the STFT is 2D). Finally, I normalized the spectrograms to the range [0–1].

*For example:
[Macaulay Library](https://www.macaulaylibrary.org/2021/07/19/from-sound-to-images-part-1-a-deep-dive-on-spectrogram-creation/)

## File Descriptions
`build_GAN.py`: Defines and tests the GAN model architecture.

`test_GAN`: Defines functions for generating simulations and ensuring the model is well defined. Also loads trained models and displays simulations.

`train.py`: Imports a dataset, defines an input pipeline, and runs a training routine.

`dataset_manager.py`: Defines a `DatasetManager` class to make it easy to switch between different datasets in `train.py`.

`make_sine_dataset.py`: Creates a mock dataset of random sine waves.

## Process
I started with a variational autoencoder (VAE), which was able to capture features in the data, but did not produce a wide variation in output, and the output was a bit of a blend between training examples, which did not sound convincing. I determined that a VAE would not be able to produce audio that sounded convincing since it's evaluated during training on how closely the output resembles the input. Consequently, it's not incentivized to be creative. For that reason, I switched to a generative adversarial network (GAN). This type of network is not directly penalized for generating something outside of the training examples, so long as it's similar enough that it can fool the discriminator, which is exactly what I wanted.

I used a small conv-net discriminator architecture for each experiment since it seemed to perform well enough and did not appear to be responsible for any lack of quality in the generator.

Generator architectures I tried:
- small (around 2-5 layers) dense
- small convolutional
- medium dense (with a couple different activation functions)
- small convolutional, with only time-domain deconvolution
- huge LSTM x LSTM (failed to train, too complex)
- LSTM + dense
- large dense

I also tried the following optimizations:
- discriminator warmup
- higher learning rate
- exponentially decaying learning rate schedule
- jittering the loss function

Additionally, I began working on a general-purpose additive synthesizer-like model which is a more parametric formulation of the problem. Contrary to the NNs, which discover order out of chaos, my parametric model aims to produce complexity from order. Using an additive approach, the model would learn where to increase complexity and add variation beginning from a simulation that is close to the training data. I hope to use unsupervised learning algorithms to discover the optimum parameters for reproducing any class of sounds. The challenges I faced designing this model were as follows:
- It's easy to accidentally make a simplifying assumption which restricts the ability of the model to learn any sound (for example, assuming the sound follows the harmonic series, or at least that overtones have rational frequency ratios; Assuming that change in frequency or amplitude over time for any sound has a single parametric formula)
- Generalizing to all possible sounds could easily involve intractably massive matrices in order to capture dependencies between various parts of the sound

In other words, the more general the model, the more complex and resource-expensive it would need to be. The more constrained the model, the less complex and faster to train, but it would be less able to generalize to any class of sound.

It seems that the model should be able to expand itself to capture arbitrarily complex relationships, but it should also be 
able to compress such relationships into the simplest form while retaining an acceptable quality threshold.

## Technologies
MockingBot was developed using the following technologies:
- Python 3.8.9
- Numpy 1.23.1
- Matplotlib 3.5.2
- TensorFlow 2.9.2

### Development Environment
The original VAE architecture was built in Google Colab (**Ubuntu Bionic Beaver**) so that I could take advantage of TensorFlow I/O's extended ability to import WAV files (the function I wanted was not supported on MacBook Air (2020, M1)), however that didn't work out, so I returned to local development on **MacOS Monterey.**

## Author
Justin Masayda [@keysmusician](https://github.com/keysmusician)
<div align="center">
<pre>
        _   _       _   _   _       _   _       _   _   _     
    ___//|_//|_____//|_//|_//|_____//|_//|_____//|_//|_//|___ 
   /  /// ///  /  /// /// ///  /  /// ///  /  /// /// ///  / |
  /  ||/ ||/  /  ||/ ||/ ||/  /  ||/ ||/  /  ||/ ||/ ||/  / / 
 /___/___/___/___/___/___/___/___/___/___/___/___/___/___/ /  
 |___|___|___|___|___|___|___|___|___|___|___|___|___|___|/   
 
</pre>
</div>

## License

All rights reserved, but feel free to ask.
