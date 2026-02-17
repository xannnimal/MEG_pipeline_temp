#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 10:40:01 2026

@author: Alexandria McPherson


C-SHARP MEG Forward & Inverse Pipeline
Supports both CTF (UCSF) and FieldLine OPM data.

Full pipeline description: 
    1. Load data (CTF vs FIF files) - make functions for each data type
    2. Do coregistration, either in line or through YORC script
    3. Preprocessing - Reject eyeblinks with ICA (?), high-pass filter, add functions for SSS, homogenous field correction, etc
    4. Make evokeds for each condition
    5. Create covariance
    6. Forward solution
    7. Inverse solutions - explore options. Most take norm and throw away the orientation. Look into “free” orientation vs locked
    8. Apply inverse to evokeds
    9. Put into BIDS structure, specifically with events
    10. Generate MNE-report

This module covers steps 1-4 of the full pipeline:
    1. Load data (CTF vs FIF files) - make functions for each data type
    2. Do coregistration, either in line or through YORC script
    3. Preprocessing - Reject eyeblinks with ICA (?), high-pass filter, add functions for SSS, homogenous field correction, etc
    4. Make evokeds for each condition - parsed with event IDs
"""

# --- Dependencies -----------------------------------------------------------
import mne
from mne import combine_evoked
from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf
from mne.minimum_norm import apply_inverse

import numpy as np
import pandas as pd
from pathlib import Path


# --- Helpers -----------------------------------------------------------------

def OPM_data(rawfile, trigger_chan, prepros_type):
    raw=mne.io.read_raw_fif(rawfile ,'default', preload=True)
    ## find events
    events = mne.find_events(raw, stim_channel=trigger_chan, shortest_event=1)
    ## always notch filter
    meg_picks = mne.pick_types(raw.info, meg=True)
    raw.notch_filter(freqs=60, picks=meg_picks)
    if prepros_type == 'high-filter':
        # # apply high-pass filter 
        freq_min = 3
        freq_max = 40
        raw.load_data().filter(l_freq=freq_min, h_freq=freq_max)
    if prepros_type == 'ssp-filter':
        freq_min = 3
        freq_max = 40
        raw.load_data().filter(l_freq=freq_min, h_freq=freq_max)
        proj = mne.compute_proj_raw(raw, start=0, stop=None, duration=1, n_grad=0, n_mag=2, n_eeg=0, reject=None, flat=None, n_jobs=None, meg='separate', verbose=None)
        raw = raw.add_proj(proj)
    # # apply notch filter for 60Hz power lines
    meg_picks = mne.pick_types(raw.info, meg=True)
    raw.notch_filter(freqs=60, picks=meg_picks)
    return raw,events


# --- Main (example usage) ---------------------------------------------------

if __name__ == '__main__':
    rawfile = Path('/Users/alexandria/Documents/STANFORD/FieldLine_tests/subjects/sub-XM/20260206_143328_sub-XM_file-xantone_raw.fif')

    ## Define constants
    trigger_chan = 'di2' # should always be 'di2' for FieldLine but could be 'di1'
    ## Load and Filter Data ## 
    ## Define filter type
    # 'high-filter' = applies high-pass filter 3, low pass 40.
    # 'ssp-filter' = applies high-pass filter 0.1, low pass 40, and SSP proj from baseline
    # more to come
    prepros_type = 'ssp-filter' 
    [raw, events] = OPM_data(rawfile, trigger_chan, prepros_type)
    
    ## Get epochs and evoked response
    tmin = -0.2  # start of each epoch (200ms before the trigger)
    tmax = 0.6  # end of each epoch (600ms after the trigger)
    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None, preload=True)
    evoked = epochs.average()
    fig = evoked.plot_joint()
    
    
    #### STEPS TO ADD
    ## save events
    # mne.write_events( participant + '/' + participant + '_events.fif',events)
    ## define triggers
    # event_id = dict(<cond1> = 1, <cond2> = 2, <cond3> = 16, <cond4> = 32)  
    ## compute covariance and write to file
    # cov = mne.cov.compute_covariance(epochs, 0)
    # cov = mne.cov.regularize(cov, evoked.info, mag=0.05, grad = 0.05, proj = True, exclude = 'bads')
    ## make foward solution
    # fwd = mne.make_forward_solution(info = info, mri = mri, src = src, bem = bem, fname = fname, meg = True, eeg = False, overwrite = True)
    ## make inverse operator 
    # inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov, loose = None, depth = None, fixed = False)
    ## apply inverse for each condition 
    
    
    
    
    
    
