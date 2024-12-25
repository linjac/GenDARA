# %%
import pandas as pd
import os
import librosa as lb
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, List, Dict

from scipy.stats import linregress as linregress
import scipy.signal as signal 
from scipy.signal import butter as butter
from scipy.signal import sosfilt as sosfilt

def get_octave_filters(center_freqs: List, sr: int, order: int = 5) -> List:
    """
    Design octave band filters (butterworth filter).
    Returns a tensor with the SOS (second order sections) representation of the filter
    """
    sos = []
    for band_idx in range(len(center_freqs)):
        center_freq = center_freqs[band_idx]
        if abs(center_freq) < 1e-6:
            # Lowpass band below lowest octave band
            f_cutoff = (1 / np.sqrt(2)) * center_freqs[band_idx + 1]
            this_sos = butter(N=order, Wn=f_cutoff, fs=sr, btype='lowpass', analog=False, output='sos')
        elif abs(center_freq - sr / 2) < 1e-6:
            f_cutoff = np.sqrt(2) * center_freqs[band_idx - 1]
            this_sos = butter(N=order, Wn=f_cutoff, fs=sr, btype='highpass', analog=False,
                                            output='sos')
        else:
            f_cutoff = center_freq * np.array([1 / np.sqrt(2), np.sqrt(2)])
            this_sos = butter(N=order, Wn=f_cutoff, fs=sr, btype='bandpass', analog=False,
                                            output='sos')

        sos.append(this_sos)

    return sos

def filter_octaveband(x, sr=16000, filter_frequencies=[125, 250, 500, 1000, 2000, 4000]):
    octave_filters = get_octave_filters(filter_frequencies, sr)
    out = []
    for sos in octave_filters:
        tmp = sosfilt(sos, x, axis=-1)
        out.append(tmp)
    out = np.stack(out, axis=1)
    # print(f'filtered signals [i, freq, t] {out.shape}')
    return out


def calculate_edf(h, 
                  sr=16000, 
                  normalize=True, 
                  per_octaveband=False, 
                  plot=False,
                  filter_frequencies=[125, 250, 500, 1000, 2000, 4000],
                  **kwargs
                  ):
    import math
    h = h[..., :math.floor(0.95*h.shape[-1])]
    if per_octaveband:
        h = filter_octaveband(h, sr=sr, filter_frequencies=filter_frequencies)
    
    energy = h**2
    edf = np.flip( np.cumsum(np.flip(energy, axis=-1), axis=-1), axis=-1)

    if normalize:
        edf /= np.max(edf, axis=-1, keepdims=True)
    edf_db = 10*np.log10(np.clip(edf,a_min=1e-8, a_max=None))
    
    if plot:
        
        for i in range(edf_db.shape[0]):
            if per_octaveband:
                for j in range(edf_db.shape[1]):
                    plt.plot(edf_db[i, j, :])
                    plt.show()
            else:
                plt.plot(edf_db[i, :])
                plt.show()
    # mean = np.mean(edf_db, axis=-1)
    
    # for i, m in enumerate(mean):
    #     if np.isnan(m) or np.isinf(m):
    #         print(np.argwhere(np.isinf(edf_db[i])))
    #         plt.plot(edf_db[i].T)
    #         plt.show()
    # Return with 5% chopped off
    # edf_db = edf_db
    return edf_db

def calculate_rt(x, 
                 sr, 
                 per_octaveband=True, 
                 filter_frequencies=[125, 250, 500, 1000, 2000, 4000], 
                 db_high_cutoff=5, 
                 db_low_cutoff=35,
                 plot=False,
                 **kwargs
                 ): # TODO must test this on broadband case
    
    edf = calculate_edf(x, sr=sr, per_octaveband=per_octaveband, filter_frequencies=filter_frequencies)
    xrange = np.arange(edf.shape[-1])/sr
    
    def rt_helper_function(edc):
        try:
            condition = (np.max(edc)-db_low_cutoff < edc) * (np.max(edc)-db_high_cutoff > edc)
            data = np.where(condition)
            i1, i2 = data[0][0], data[0][-1]
            result = linregress(xrange[i1:i2], edc[i1:i2])
            rt = -60/result.slope
        except:
            rt = np.nan
            return rt, []
        return rt, result.slope*np.arange(len(edc)) + result.intercept
    
    params = []
    for i in range(edf.shape[0]):
        if len(edf.shape) == 3:
            param_octaveband = []
            for j in range(edf.shape[1]):
                edc =  edf[i, j, :]
                rt, lin_line = rt_helper_function(edf[i, j, :])
                if plot:
                    plt.plot(edc)
                    # plt.plot(lin_line)
                    # plt.ylim(-40, 0)
                    plt.show()
                param_octaveband.append(rt)
            params.append(param_octaveband)
        else:
            edc = edf[i, :]        
            rt, lin_line = rt_helper_function(edf[i, :])
            if plot:
                plt.plot(edc)
                # plt.plot(lin_lin e)
                # plt.ylim(-40, 0)
                plt.show()
            params.append(rt)
        
    params = np.array(params)
    return params

def calculate_drr(rir, win_len=0.001, sr=16000, **kwargs):
    rir = torch.Tensor(rir)
    nd = torch.argmax(torch.abs(rir), axis=-1)
    n0 = torch.tensor(round(win_len*sr))
    mask1 = nd > n0
    nd = nd * mask1
    
    max_len = rir.size(-1)
    mask = torch.arange(max_len).expand(rir.size(0), max_len) 
    mask = (mask < (nd + n0).view(-1,1))
    if rir.dim() == 3: # process batched data
        mask = mask.unsqueeze(1)
    
    energy = rir**2
    direct = energy * mask
    reverb = energy * ~mask

    drr = 10 * torch.log10(torch.sum(direct, axis=-1)/
                        torch.sum(reverb, axis=-1))
    return drr.numpy()

def load_rirs(rirdir, meta_filepath):
    metadata = pd.read_csv(meta_filepath)
    dataset = []
    for filename in metadata['filename']: # TODO full_filename
        rir, sr = lb.load(os.path.join(rirdir, filename), sr=None)
        dataset.append(rir)
    dataset = np.array(dataset)
    print(f'Loaded dataset shape: {dataset.shape}')
    return dataset

def make_dataset_metadata(dataset_dir):
    raise NotImplementedError
    filename = []
    dist_gt = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.wav'):
                filename.append(file)
                dist_gt.append(float(file.split('_')[2][:-1])) # TODO  Make sure the naming convention is correct
    df = pd.DataFrame({'filename': filename, 'dist_gt': dist_gt})
    df.to_csv(os.path.join(dataset_dir, 'meta.csv'))
    
# %%
