import numpy as np
from scipy import signal
from scipy.signal import find_peaks

def bandpass_filter(signal_data, fs, low, high):
    nyq = 0.5 * fs
    low, high = low / nyq, high / nyq
    b, a = signal.butter(2, [low, high], btype='band')
    return signal.filtfilt(b, a, np.array(signal_data))

def calculate_bpm(filtered_signal, fps):
    peaks, _ = find_peaks(filtered_signal, distance=int(fps * 0.5), height=np.percentile(filtered_signal, 60))
    if len(peaks) > 1:
        peak_times = np.diff(peaks) / fps
        bpm = 60 / np.mean(peak_times)
        if 40 <= bpm <= 180:
            return bpm
    return None


# import numpy as np
# from scipy import signal
# from scipy.signal import find_peaks, medfilt
#
#
# def adaptive_bandpass_filter(signal_data, fs):
#     """ Adaptive bandpass filter that adjusts based on signal properties. """
#     nyq = 0.5 * fs
#     low, high = 0.7, 3.0  # Default range for heart rate (40–180 BPM)
#
#     # Adaptive frequency selection (optional, can be refined)
#     signal_std = np.std(signal_data)
#     if signal_std > 1.5:  # Example threshold (tune as needed)
#         low, high = 0.5, 3.5  # Adjust for noisy signals
#
#     low, high = low / nyq, high / nyq
#     b, a = signal.butter(2, [low, high], btype='band')
#     return signal.filtfilt(b, a, np.array(signal_data))
#
#
# def calculate_bpm(filtered_signal, fps):
#     """ Calculates BPM using peak detection with adaptive thresholding and outlier removal. """
#     # Adaptive threshold based on signal intensity
#     threshold = np.percentile(filtered_signal, 60) + np.std(filtered_signal) * 0.1
#
#     peaks, _ = find_peaks(filtered_signal,
#                           distance=int(fps * 0.5),
#                           height=threshold)
#
#     if len(peaks) > 1:
#         peak_times = np.diff(peaks) / fps
#         bpm_values = 60 / peak_times
#
#         # Apply median filtering to remove outliers
#         bpm_values = medfilt(bpm_values, kernel_size=3)
#
#         # Return median BPM if within realistic range
#         bpm = np.median(bpm_values)
#         if 40 <= bpm <= 180:
#             return bpm
#     return None