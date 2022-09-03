# MockingBot
MockingBot is a generative sound effect model which mimics the sound of whatever it is trained on, creating original audio similar to the training data.

# Process
Architectures I tried:
    small dense
    small convolutional
    medium dense (with a couple different activation functions)
    small convolutional, with only time-domain deconvolution
    huge LSTM x LSTM (failed to train, too complex)
    LSTM + dense
    large dense

I also tried the following optimizations:
    discriminator warmup
    higher learning rate
    exponentially decaying learning rate schedule
    jittering the loss function

Additionally, I began working on a general-purpose additive synthesizer-like model which is a more parametric formulation of the problem. Contrary to the NNs, which discover order out of chaos, my parametric model aims to produce complexity from order using an additive approach. I hope to use unsupervised learning algorithms to discover the optimum parameters for reproducing any class of sounds.

# Input pipeline
As others* have done when training audio models, trained over the real component of the short-time fourier transform of the signal rather than directly over the time-domain signals. This has the advantage of being phase-invariant as well as enabling convolutional architectures, since the STFT is 2D.

*For example:
[Macaulay Library](https://www.macaulaylibrary.org/2021/07/19/from-sound-to-images-part-1-a-deep-dive-on-spectrogram-creation/)


