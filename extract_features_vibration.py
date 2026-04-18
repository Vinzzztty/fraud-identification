#Extract features from vibration signals

from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis


from scipy.signal import butter, filtfilt, hilbert
import numpy as np

def fft_filtered_envelope(signal, sampling_rate, max_freq=300):

    n = len(signal)

    # -------------------------
    # Bandpass filter (1000–4000 Hz)
    # -------------------------
    fcut_low = 1000
    fcut_high = 4000
    order = 4

    nyquist = sampling_rate / 2
    low = fcut_low / nyquist
    high = fcut_high / nyquist

    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)

    # hilbert transform to get the envelope
    analytic_signal = hilbert(filtered)
    envelope = np.abs(analytic_signal)

    #hanning window  
    window = np.hanning(n)
    envelope_windowed = envelope * window

    #ffft 
    fft_vals = np.fft.rfft(envelope_windowed)
    magnitude = np.abs(fft_vals) * (2.0 / n)
    freq = np.fft.rfftfreq(n, 1/sampling_rate)

    #keep low frequencies 
    mask = freq <= max_freq
    freq = freq[mask]
    magnitude = magnitude[mask]

    db = 20 * np.log10(magnitude + 1e-6)  # Convert to dB, add small value to avoid log(0)

    return freq, db, magnitude

def find_fundamental_freq(freq, db, rpm):
    bandwidth = 1  # Hz
    fr_calculate = rpm / 60

    # ambil region sekitar fundamental
    mask = (freq >= fr_calculate - bandwidth) & (freq <= fr_calculate + bandwidth)

    # ambil data dalam window
    freq_window = freq[mask]
    db_window = db[mask]

    # cari peak (amplitudo terbesar)
    idx_peak = np.argmax(db_window)

    fr = freq_window[idx_peak]

    return fr

def bearing_fault_frequencies(fr, n, Bd, Pd, theta=0):

    theta = np.deg2rad(theta)

    BPFO = (n/2) * fr * (1 - (Bd/Pd)*np.cos(theta))
    BPFI = (n/2) * fr * (1 + (Bd/Pd)*np.cos(theta))
    BSF  = (Pd/(2*Bd)) * fr * (1 - ((Bd/Pd)*np.cos(theta))**2)
    FTF  = 0.5 * fr * (1 - (Bd/Pd)*np.cos(theta))

    return BPFO, BPFI, BSF, FTF


def extract_time_domain_features(signal):

    signal = np.array(signal)

    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms
    skewness = skew(signal)
    kurt = kurtosis(signal, fisher=False)

    return rms, peak, crest_factor, skewness, kurt

def amplitude_near(freq, values, target_freq):

    bandwidth = 1  # Hz

    mask = (freq >= target_freq - bandwidth) & (freq <= target_freq + bandwidth)

    if np.any(mask):
        return np.max(values[mask])
    else:
        return -120  # very low dB if not found
    
def amplitude_ratio(freq, magnitude, target_freq):

    peak_bw = 0.5 # bandwidth peak
    guard_bw = 5     # zona aman (hindari leakage)
    noise_bw = 10    # area noise

    # --- PEAK ---
    mask_peak = (freq >= target_freq - peak_bw) & (freq <= target_freq + peak_bw)

    if not np.any(mask_peak):
        return 0

    peak = np.max(magnitude[mask_peak])

    # --- NOISE LEFT ---
    mask_left = (freq >= target_freq - guard_bw - noise_bw) & \
                (freq <= target_freq - guard_bw)

    # --- NOISE RIGHT ---
    mask_right = (freq >= target_freq + guard_bw) & \
                 (freq <= target_freq + guard_bw + noise_bw)

    noise_values = []

    if np.any(mask_left):
        noise_values.extend(magnitude[mask_left])

    if np.any(mask_right):
        noise_values.extend(magnitude[mask_right])

    if len(noise_values) == 0:
        noise = np.median(magnitude)
    else:
        noise = np.median(noise_values)

    return peak / (noise + 1e-12)

def extract_frequency_domain_features(signal, rpm, n_ball, Bd, Pd, sampling_rate):

    freq, db, magnitude = fft_filtered_envelope(signal, sampling_rate)

    psd = magnitude**2
    psd_norm = psd / (np.sum(psd) + 1e-12)

    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
    # spectral_centroid = np.sum(freq * psd) / (np.sum(psd) + 1e-12)
    # spectral_kurtosis = kurtosis(psd)

    spectral_centroid = np.sum(freq * magnitude) / (np.sum(magnitude) + 1e-12)
    spectral_kurtosis = kurtosis(magnitude)

    # -------------------------
    # Bearing frequencies
    # -------------------------

    freq_fundamental = find_fundamental_freq(freq, db, rpm)
    
    BPFO, BPFI, BSF, FTF = bearing_fault_frequencies(freq_fundamental, n_ball, Bd, Pd)

    bpfo_ratio = amplitude_ratio(freq, magnitude, BPFO)
    bpfo_2_ratio = amplitude_ratio(freq, magnitude, 2 * BPFO)
    bpfo_3_ratio = amplitude_ratio(freq, magnitude, 3 * BPFO)
    bpfi_ratio = amplitude_ratio(freq, magnitude, BPFI)
    bsf_ratio = amplitude_ratio(freq, magnitude, BSF)
    bsf_2_ratio = amplitude_ratio(freq, magnitude, 2 * BSF)
    ftf_ratio = amplitude_ratio(freq, magnitude, FTF)

    bpfo_db = (amplitude_near(freq, db, BPFO) + amplitude_near(freq, db, 3 * BPFO)) / 2
    bpfi_db = (amplitude_near(freq, db, BPFI) + amplitude_near(freq, db, 3 * BPFI)) / 2
    bsf_db = (amplitude_near(freq, db, BSF) + amplitude_near(freq, db, 3 * BSF)) / 2
    ftf_db = (amplitude_near(freq, db, FTF) + amplitude_near(freq, db, 3 * FTF)) / 2

    return (
        spectral_entropy,
        spectral_centroid,
        spectral_kurtosis,
        bpfo_db,
        bpfi_db,
        bsf_db,
        ftf_db, 
        bpfo_ratio,
        bpfi_ratio,
        bsf_ratio,
        ftf_ratio
    )

def extract_features(signal, rpm, n_ball, Bd, Pd, sampling_rate):
    time_features = extract_time_domain_features(signal)
    freq_features = extract_frequency_domain_features(signal, rpm, n_ball, Bd, Pd, sampling_rate)

    features = np.concatenate([time_features, freq_features])

    return features