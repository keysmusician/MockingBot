'''
Attempts to find repetition schemes across various time scales.
'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
import numpy as np
import tensorflow as tf


tau = np.pi * 2

sample_rate = 44_100

'''
audio, sample_rate = tf.audio.decode_wav(
    contents=tf.io.read_file('ABAC.wav'),
    desired_channels=1
)

for window_end in range(len(audio[:1000])):
    if window_end % 50 == 0:
    # do wrappy stuff

        #plt.polar(np.linspace(0, tau, window_end), audio[:window_end])
        #plt.show()
        pass


wavelength_of_frequency = lambda Hz: sample_rate / Hz # unit: samples
frequency_of_wavelength = lambda samples: sample_rate / samples # unit: Hz
'''
time_start, time_stop = 0, 500 # unit: samples

time_steps = np.arange(time_start, time_stop)

wave_frequency = 100

audio = np.sin(tau * wave_frequency * time_steps / sample_rate)

signal = lambda time: np.sin(tau * wave_frequency * time / sample_rate)

figure = plt.figure()

gridspec = GridSpec(2,2)

cartesian = figure.add_subplot(gridspec[0, :])

cartesian.plot(time_steps, audio)

polar_pos = plt.subplot(212, projection='polar')

polar_neg = plt.subplot(212, projection='polar')

window_resolution = 500

theta = np.linspace(0, tau, window_resolution)

wrap_frequency_resolution = 500

wrap_frequency_min, wrap_frequency_max = 0, 200

for zeta in np.linspace(
        wrap_frequency_min, wrap_frequency_max, wrap_frequency_resolution):

    polar.cla()

    polar.plot(theta, signal(zeta * theta))

    polar.plot(theta, signal(zeta * theta))

    polar.set_rmin(0)

    plt.pause(0.001)


plt.show()
