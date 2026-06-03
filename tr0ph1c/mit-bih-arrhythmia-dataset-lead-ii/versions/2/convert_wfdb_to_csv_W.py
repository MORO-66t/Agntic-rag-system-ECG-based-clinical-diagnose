import wfdb
import pandas as pd
import numpy as np
from pathlib import Path
# import modules.preprocessor as preprocessor


from scipy.signal import resample

TARGET_LEN = 187
TARGET_FS = 125
mitbih_base_path = Path(__file__).resolve().parent / 'mit-bih-arrhythmia-database-1.0.0'


def record_path(record_name: str) -> str:
    return str(mitbih_base_path / record_name)

def convert_to_target_fs(signal: np.ndarray, original_fs: int) -> np.ndarray:
    """
    Convert beat from original fs to 125Hz
    """
    if original_fs == TARGET_FS:
        return signal

    new_len = int(len(signal) * TARGET_FS / original_fs)
    return resample(signal, new_len)


def normalize_beat_length(signal: np.ndarray) -> np.ndarray:
    """
    Make beat exactly 187 samples
    """
    signal = np.asarray(signal, dtype=np.float32)

    if len(signal) < TARGET_LEN:
        signal = np.pad(
            signal,
            (0, TARGET_LEN - len(signal)),
            mode='constant',
            constant_values=0
        )

    elif len(signal) > TARGET_LEN:
        signal = resample(signal, TARGET_LEN)

    return signal


def peak_to_dict(
    signal_lead: np.ndarray,
    prev_r: int,
    curr_r: int,
    next_r: int,
    rr_interval: np.int16,
    label: str,
    fs: int
) -> dict:

    # take full beat between prev and next R
    beat_signal = signal_lead[prev_r:next_r]

    # convert to 125 Hz
    beat_signal = convert_to_target_fs(beat_signal, fs)

    # force to 187 samples
    beat_signal = normalize_beat_length(beat_signal)

    peak_dict = {}

    for i, val in enumerate(beat_signal):
        peak_dict[f'samp_{i}'] = val

    peak_dict['rr_interval'] = rr_interval
    peak_dict['label'] = label

    return peak_dict

def get_rr_intervals(annon_samples: np.ndarray, fs: int) -> np.ndarray:
    rr_samples = np.diff(annon_samples)
    rr_intervals = rr_samples / fs * 1000  
    return rr_intervals

def is_beat_annotation(symbol: str) -> bool:
    beat_symbols = {'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'}
    return symbol in beat_symbols

def is_file_exists(file_path: str) -> bool:
    try:
        with open(file_path, 'r'):
            return True
    except FileNotFoundError:
        return False
    
def shoud_skip_last_peak(last_r_signal_index, last_peak_index) -> bool:
    return last_r_signal_index + 45 > last_peak_index

def filter_beat_samples(annotations: list[str], samples: list[int]) -> list[int]:
    return [sample for ann, sample in zip(annotations, samples) if is_beat_annotation(ann)]

def get_record_peaks(record_name: str) -> pd.DataFrame:

    record = wfdb.rdrecord(record_path(record_name))
    record_annon = wfdb.rdann(
        record_path(record_name),
        'atr',
        return_label_elements=['symbol', 'label_store']
    )

    # filter beat annotations only
    beat_symbols = []
    beat_samples = []

    for sym, samp in zip(record_annon.symbol, record_annon.sample):
        if is_beat_annotation(sym):
            beat_symbols.append(sym)
            beat_samples.append(samp)

    r_samples = np.array(beat_samples)
    rr_intervals = get_rr_intervals(r_samples, record.fs)

    # Lead II
    lead2 = record.p_signal[:, 0]
    print(record.sig_name)

    peaks_data = []

    # skip first and last because we need prev and next R
    for i in range(1, len(r_samples) - 1):

        prev_r = r_samples[i - 1]
        curr_r = r_samples[i]
        next_r = r_samples[i + 1]

        rr_interval = rr_intervals[i - 1]
        label = beat_symbols[i]

        peak_dict = peak_to_dict(
            lead2,
            prev_r,
            curr_r,
            next_r,
            rr_interval,
            label,
            record.fs
        )

        peaks_data.append(peak_dict)

    return pd.DataFrame(peaks_data)
def convert_wfdb_to_csv(output_csv: str, record_names: list[str]):
    all_peaks_df = pd.DataFrame()
    
    for i, record_name in enumerate(record_names):
        print(f'Processing record {i+1}/{len(record_names)}: {record_name}')
        record_peaks_df = get_record_peaks(record_name)
        all_peaks_df = pd.concat([all_peaks_df, record_peaks_df], ignore_index=True)
    
    all_peaks_df.to_csv(output_csv, index=False)

def is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

# def get_files_unique_names(directory_path):

#     dir_path = Path(directory_path)

#     unique_names = {file.stem for file in dir_path.iterdir() if file.is_file() and is_numeric(file.stem)}

#     return list(unique_names)
def get_files_unique_names(directory_path):
    dir_path = Path(directory_path)
    unique_names = {file.stem for file in dir_path.iterdir()
                    if file.is_file() and is_numeric(file.stem)}
    return sorted(list(unique_names))



if __name__ == '__main__':
    convert_wfdb_to_csv('mitbih_beats1.csv', get_files_unique_names(mitbih_base_path))