import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import copy


class Chebyshev():
    def __init__(self, order, ripple, critical_freq, btype, sampling_freq, forward_backward=True):

        self.b, self.a = signal.cheby1(N=order, rp=ripple, Wn=critical_freq, btype=btype, output='ba', fs=sampling_freq)
        self.forward_backward = forward_backward
        self.sample_f = sampling_freq
        self.critical_freq = critical_freq
        self.ripple = ripple

    def apply(self, X):
        if self.forward_backward:
            return signal.filtfilt(self.b, self.a, X)

        else:
            return signal.lfilter(self.b, self.a, X)

    def check(self):

        fig, ax1 = plt.subplots()
        ax1.set_title('Digital filter frequency response')
        w, h = signal.freqz(self.b, self.a)
        w = w / np.pi * self.sample_f / 2.0

        ax1.plot(w, 20 * np.log10(abs(h)), 'b')
        ax1.set_ylabel('Amplitude [dB]', color='b')
        ax1.set_xlabel('Frequency [Hz]')

        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w, angles, 'g')
        ax2.set_ylabel('Angle (radians)', color='g')
        ax2.grid()
        ax2.axis('tight')
        plt.show()


def filter_bank(data, filter_banks = [(4, 16), (16, 40)]):
    EEG_bank = []
    for bank in filter_banks:
        parameter = signal.butter(N=5, Wn=bank, btype='bandpass', fs=250)
        EEG_filtered = signal.lfilter(parameter[0], parameter[1], data)[...,
                       0 * 250::1]
        EEG_bank.append(EEG_filtered)
    data = np.concatenate(EEG_bank, axis=1)

    return data


def filter_bank_v2(data, filter_banks = [(4, 16), (16, 40)]):
    original_data = copy.deepcopy(data)
    EEG_bank = []
    for bank in filter_banks:
        parameter = signal.butter(N=5, Wn=bank, btype='bandpass', fs=250)
        EEG_filtered = signal.lfilter(parameter[0], parameter[1], data)[...,
                       0 * 250::1]
        EEG_bank.append(EEG_filtered)
    data = np.concatenate(EEG_bank, axis=1)
    data = np.concatenate((data, original_data), axis=1)

    return data