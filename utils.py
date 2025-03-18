import scipy.io as sio
import random
import os
import torch
import torch.nn as nn


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def init_weights_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

def create_SMC(frequency, len_, dt, PW, base_amplitude, amplitude):
    """
    Create SMC (Spinal Motor Command) signal.

    Parameters:
    frequency (float): Frequency of the SMC signal.
    len_ (int): Length of the SMC signal.
    dt (float): Time step.
    PW (float): Pulse width.
    base_amplitude (float): Base amplitude of the SMC signal.
    amplitude (float): Amplitude of the SMC signal.

    Returns:
    np.ndarray: SMC signal.
    """

    # Define time vector
    pulse_w = int(PW / dt)  # Pulse width in index

    # Create initial SMC array and pulse array
    SMC = np.zeros(len_)
    iD = amplitude - base_amplitude  # nA/cm2 [SMC current amplitude]
    pulse = np.full(pulse_w, iD)

    # Loop through SMC array and embed pulse array at random intervals
    impulse = np.zeros(frequency)  # List of impulse locations
    inst = frequency
    isi = 1000 / inst
    ipi = round(isi * 1 / dt)
    i = 0
    n_im = 0

    while i < len_:
        j = np.random.randint(i + int(ipi * 0.5), i + ipi + 1)
        if j > len_:
            break
        SMC[j:j + pulse_w] = pulse
        i += round(isi * 1 / dt)
        impulse[n_im] = j
        n_im += 1

    # Replace 0s with base_amplitude
    SMC += base_amplitude

    return SMC


def rl_BGM_reset_SMC_pulse_python_step(tmax, IT, pd, corstim, freq, pprofile, kk, len, SMC_pulse, dt, sliding):
    random.seed()  # set seed for reproducibility

    n = 10  # number of neurons in each nucleus
    t = np.arange(0, tmax + dt, dt)

    # DBS Parameters
    PW = 0.3  # ms [DBS pulse width]
    amplitude = 300  # nA/cm2 [DBS current amplitude]

    if isinstance(freq, str):
        pattern = random.randint(0, 200)
    else:
        pattern = freq

    # Create DBS Current
    if pattern == 0:
        Idbs = np.zeros(len(t))
    else:
        Idbs, impulse = creatdbs(pattern, tmax, dt, PW, amplitude, pprofile)

    # Create Cortical Stimulus Pulse
    Iappco = np.zeros(len(t))
    if corstim == 1:
        Iappco[int(1000 / dt):int((1000 + 0.3) / dt)] = 350

    # Run CTX-BG-TH Network Model
    vgpi, vTH, CTX_workspace = CTX_BG_TH_network(pd, corstim, tmax, dt, n, Idbs, Iappco, SMC_pulse)

    # Calculate GPi pathological low-frequency oscillatory power
    dt1 = dt * 1e-3
    params = {
        'Fs': 1 / dt1,
        'fpass': [1, 100],
        'tapers': [3, 5],
        'trialave': 1
    }

    # Assuming sliding_window_2 and Error_Index are already defined
    beta_vec, EI, kk = sliding_window_2(vgpi, vTH, SMC_pulse, params, kk, n, len, CTX_workspace['timespike'], dt,
                                        sliding)

    # Save results
    name = f"{IT}con{pattern}rs.mat" if pd == 0 else f"{IT}pd{pattern}rs.mat"
    sio.savemat(name, {'beta_vec': beta_vec, 'EI': EI, 'Idbs': Idbs})


def creatdbs(pattern, tmax, dt, PW, amplitude, pprofile):
    # Define time vector
    t = np.arange(0, tmax + dt, dt)
    pulse_w = int(PW / dt)  # Pulse width in index

    # Create initial DBS array and pulse array (iD is the amplitude)
    Idbs = np.zeros(len(t))
    iD = amplitude  # nA/cm2 [DBS current amplitude]
    pulse = np.ones(pulse_w) * iD

    # Loop through Idbs array and embed pulse array at random intervals
    if pprofile is None:
        impulse = np.zeros(pattern)  # list of impulse locations
        inst = pattern
        isi = 1000 / inst
        ipi = int(round(isi / dt))
        i = 0
        n_im = 0
        while i < len(t):
            j = np.random.randint(i, i + ipi + 1)
            if j >= len(t):
                break
            Idbs[j:j + pulse_w] = pulse
            i += int(round(isi / dt))
            impulse[n_im] = j
            n_im += 1
    else:
        impulse = np.where(pprofile)[0]
        if len(pprofile) != len(Idbs):
            impulse = impulse * int(round(len(Idbs) / len(pprofile)))

        if impulse[-1] > len(Idbs) - 30:
            impulse[-1] = len(Idbs) - 30

        for i in impulse:
            Idbs[i:i + pulse_w] = pulse

    return Idbs, impulse


def sliding_window_2(v, vth, SMC, params, kk, n, len_, timespike, dt, slide_size):
    vpi_len = v.shape[1]
    beta_vec = np.zeros(len_)
    error_index = np.zeros(len_)
    time = np.arange(0, len_ + 1, dt)
    j = 1

    while j <= len_:
        beginning = int(vpi_len - len_ / dt - (len_ - j) / dt)
        ending = int(vpi_len - (len_ - j) / dt)

        vpi_1 = v[:, beginning:ending]
        vth_1 = vth[:, beginning:ending]

        GPi_APs = find_spike_times(vpi_1, time, n)
        GPi_area, _, _ = make_Spectrum(GPi_APs, params)
        beta_vec[j-1] = GPi_area

        t_hold = np.arange(beginning * dt, ending * dt + dt, dt)
        ts_1 = timespike[(timespike > t_hold[0]) & (timespike < t_hold[-1])]

        error_index[j-1] = calculateEI(t_hold, vth_1, ts_1, t_hold[0], t_hold[-1])

        kk += 100
        j += slide_size

    beta_vec = beta_vec[np.nonzero(beta_vec)]
    error_index = error_index[np.nonzero(error_index)]

    return beta_vec, error_index, kk


def calculateEI(t, vth, timespike, tstart, tend):
    m = vth.shape[0]
    e = np.zeros(m)
    b1 = np.where(timespike >= tstart)[0][0]
    b2 = np.where(timespike <= tend - 25)[0][-1]

    for i in range(m):
        compare = []
        for j in range(1, len(vth[i])):
            if vth[i, j-1] < -40 and vth[i, j] > -40:
                compare.append(t[j])

        for p in range(b1, b2 + 1):
            if p != b2:
                a = [x for x in compare if x >= timespike[p] and x < timespike[p] + 25]
                b = [x for x in compare if x >= timespike[p] + 25 and x < timespike[p + 1]]
            else:
                a = [x for x in compare if x >= timespike[p] and x < tend]
                b = []

            if len(a) == 0 or len(a) > 1:
                e[i] += 1
            if len(b) > 0:
                e[i] += len(b)

    er = np.mean(e / (b2 - b1 + 1))
    return er


def PSD_calc(data, params, n):
    fs = params['Fs']
    if len(data) % 2 != 0:
        data = data[:, 1:]
    fftdata = np.fft.fft(data, axis=1)
    fftdata = fftdata[:, :fftdata.shape[1] // 2 + 1]
    psddata = (1 / (fs * len(fftdata))) * np.abs(fftdata) ** 2
    psddata[:, 1:-1] *= 2
    w = np.linspace(0, fs / 2, len(psddata[0]))
    windowfft = psddata[:, (w > 13) & (w < 35)]
    int_ = np.trapz(windowfft, axis=1, dx=(w[1] - w[0]))
    SS = np.mean(psddata, axis=0)
    SS = np.mean(SS[(w > 13) & (w < 35)])
    beta = np.mean(int_)
    return beta, w, SS


def EI_calc(voltage, pulse, n, dt):
    lookup_length = 40
    TH_pulse_height = -40
    end_length = -1
    SMC_pulse = 0
    TH_miss = 0

    for i in range(len(voltage)):
        if pulse[i] < 0.1 or i < end_length:
            continue
        SMC_pulse += 1
        end_length = i + int(lookup_length / dt)
        if end_length > len(voltage):
            end_length = len(voltage)

        for j in range(n):
            if max(voltage[j, i:end_length]) < TH_pulse_height:
                TH_miss += 1

    EI = TH_miss / (n * SMC_pulse)
    return EI


def find_spike_times(v, t, nn):
    data = [None] * nn
    t = t / 1000  # Convert to seconds
    for k in range(nn):
        data[k] = {'times': t[np.diff(v[k, :] > -20) == 1]}
    return data


def gpe_ainf(V):
    return 1 / (1 + np.exp(-(V + 57) / 2))

def gpe_hinf(V):
    return 1 / (1 + np.exp((V + 58) / 12))

def gpe_minf(V):
    return 1 / (1 + np.exp(-(V + 37) / 10))

def gpe_ninf(V):
    return 1 / (1 + np.exp(-(V + 50) / 14))

def gpe_rinf(V):
    return 1 / (1 + np.exp((V + 70) / 2))

def gpe_sinf(V):
    return 1 / (1 + np.exp(-(V + 35) / 2))

def gpe_tauh(V):
    return 0.05 + 0.27 / (1 + np.exp(-(V + 40) / -12))

def gpe_taun(V):
    return 0.05 + 0.27 / (1 + np.exp(-(V + 40) / -12))

def th_hinf(V):
    return 1 / (1 + np.exp((V + 41) / 4))

def th_minf(V):
    return 1 / (1 + np.exp(-(V + 37) / 7))

def th_pinf(V):
    return 1 / (1 + np.exp(-(V + 60) / 6.2))

def th_rinf(V):
    return 1 / (1 + np.exp((V + 84) / 4))

def th_tauh(V):
    return 1 / (ah(V) + bh(V))  # Assuming ah(V) and bh(V) are defined elsewhere

import numpy as np

# Functions for Thalamus
def ah(V):
    return 0.128 * np.exp(-(V + 46) / 18)

def bh(V):
    return 4 / (1 + np.exp(-(V + 23) / 5))

def th_taur(V):
    return 0.15 * (28 + np.exp(-(V + 25) / 10.5))

# Alpha and Beta Functions
def alphah(V):
    return 0.128 * np.exp((-50 - V) / 18)

def alpham(V):
    return (0.32 * (54 + V)) / (1 - np.exp((-54 - V) / 4))

def alphan(V):
    return (0.032 * (52 + V)) / (1 - np.exp((-52 - V) / 5))

def alphap(V):
    return (3.209e-4 * (30 + V)) / (1 - np.exp((-30 - V) / 9))

def betah(V):
    return 4 / (1 + np.exp((-27 - V) / 5))

def betan(V):
    return 0.5 * np.exp((-57 - V) / 40)

def betam(V):
    return 0.28 * (V + 27) / ((np.exp((27 + V) / 5)) - 1)

def betap(V):
    return (-3.209e-4 * (30 + V)) / (1 - np.exp((30 + V) / 9))

# GABA Function
def Ggaba(V):
    return 2 * (1 + np.tanh(V / 4))

# Functions for STN
def stn_ainf(V):
    return 1 / (1 + np.exp(-(V + 45) / 14.7))

def stn_binf(V):
    return 1 / (1 + np.exp((V + 90) / 7.5))

def stn_cinf(V):
    return 1 / (1 + np.exp(-(V + 30.6) / 5))

def stn_d1inf(V):
    return 1 / (1 + np.exp((V + 60) / 7.5))

def stn_d2inf(V):
    return 1 / (1 + np.exp((V - 0.1) / 0.02))

def stn_hinf(V):
    return 1 / (1 + np.exp((V + 45.5) / 6.4))

def stn_minf(V):
    return 1 / (1 + np.exp(-(V + 40) / 8))

def stn_ninf(V):
    return 1 / (1 + np.exp(-(V + 41) / 14))

def stn_pinf(V):
    return 1 / (1 + np.exp(-(V + 56) / 6.7))

def stn_qinf(V):
    return 1 / (1 + np.exp((V + 85) / 5.8))

def stn_rinf(V):
    return 1 / (1 + np.exp(-(V - 0.17) / 0.08))

# Tau Functions for STN
def stn_taua(V):
    return 1 + 1 / (1 + np.exp(-(V + 40) / -0.5))

def stn_taub(V):
    return 200 / (np.exp(-(V + 60) / -30) + np.exp(-(V + 40) / 10))

def stn_tauc(V):
    return 45 + 10 / (np.exp(-(V + 27) / -20) + np.exp(-(V + 50) / 15))

def stn_taud1(V):
    return 400 + 500 / (np.exp(-(V + 40) / -15) + np.exp(-(V + 20) / 20))

def stn_tauh(V):
    return 24.5 / (np.exp(-(V + 50) / -15) + np.exp(-(V + 50) / 16))

def stn_taum(V):
    return 0.2 + 3 / (1 + np.exp(-(V + 53) / -0.7))

def stn_taun(V):
    return 11 / (np.exp(-(V + 40) / -40) + np.exp(-(V + 40) / 50))

def stn_taup(V):
    return 5 + 0.33 / (np.exp(-(V + 27) / -10) + np.exp(-(V + 102) / 15))

def stn_tauq(V):
    return 400 / (np.exp(-(V + 50) / -15) + np.exp(-(V + 50) / 16))


# Spectrum Function
def make_Spectrum(raw, params):
    from scipy.signal import spectrogram
    f, _, S = spectrogram(raw, fs=params['Fs'], nperseg=params['nperseg'])
    beta = S[(f > 13) & (f < 35)]
    betaf = f[(f > 13) & (f < 35)]
    area = np.trapz(betaf, beta)
    return area, S, f
