# Various chirp preprocessing functions to highlight certain features or
# provide the input that Tensorflow expects.

import numpy as np

def raw_normalize(data):  # Minmax normalization
    normalized_data = (data - data.min(0)) / data.ptp(0)

    return normalized_data

def normalize(data):  # z-score normalization
    normalized_data = (data - np.mean(data)) / np.std(data)

    return normalized_data

def ifreq(data, sampling_freq):
    instantaneous_phase = np.unwrap(np.angle(data))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * sampling_freq)
    normalized_instantaneous_frequency = normalize(instantaneous_frequency)

    # Append the last value of the ifreq to the array, so that the size of the tensor stays a power of 2.
    normalized_instantaneous_frequency = np.append(normalized_instantaneous_frequency, normalized_instantaneous_frequency[-1])

    return normalized_instantaneous_frequency

def ifreq_nonorm(data, sampling_freq):
    instantaneous_phase = np.unwrap(np.angle(data))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * sampling_freq)

    return instantaneous_frequency

def iphase(data):
    instantaneous_phase = np.unwrap(np.angle(data))
    normalized_instantaneous_phase = normalize(instantaneous_phase)

    return normalized_instantaneous_phase

def iphase_nonorm(data):
    instantaneous_phase = np.unwrap(np.angle(data))

    return instantaneous_phase

def iphase_nonorm_wrapped(data):
    instantaneous_phase = np.angle(data)

    return instantaneous_phase

def iamp(data):
    instantaneous_amplitude = np.abs(data)
    normalized_instantaneous_amplitude = normalize(instantaneous_amplitude)

    return normalized_instantaneous_amplitude

def fft(data):
    fourier = np.fft.fft(data)
    fourier_m = np.abs(fourier)

    fourier_mnorm = normalize(fourier_m)

    return fourier_mnorm

def _gradient_absolute(chirp, sample_rate):
    instantaneous_phase = np.unwrap(np.angle(chirp))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * sample_rate)
    gradient = np.gradient(instantaneous_frequency)
    return gradient

def _gradient(chirp, sample_rate=0):
    instantaneous_phase = np.unwrap(np.angle(chirp))
    instantaneous_frequency = np.diff(instantaneous_phase)  # Constant multiplications can be dropped
    gradient = np.gradient(instantaneous_frequency)
    return gradient

def roll_to_base_old(chirp, sample_rate, sf=7):
    g = _gradient(chirp, sample_rate)
    to_roll = -np.argmin(g)
    return np.roll(chirp, to_roll)

def roll_to_base(chirp, sf=7):
    # Find out the number of samples per chip
    num_chips = 2**sf
    num_samples = len(chirp)
    num_samples_per_chip = num_samples / num_chips

    # Calculate gradient
    g = _gradient(chirp)

    # Decimate gradient according to samples per chip
    n = num_samples_per_chip
    g_decim = [np.mean(g[i:i+n]) for i in range(0, len(g), n)]
    assert(len(g_decim) == num_samples / num_samples_per_chip)

    # Get the minimum gradient chip
    g_chip_min = np.argmin(g_decim)

    #print("G: " + str(min(g_decim)) + " (" + str(g_chip_min) + ")")

    # Roll to base
    to_roll = -(g_chip_min * num_samples_per_chip)
    return np.roll(chirp, to_roll)
