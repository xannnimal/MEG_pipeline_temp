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
import os
from mne.io import read_raw_ctf
from mne.minimum_norm import apply_inverse

import numpy as np
import pandas as pd
from pathlib import Path


# --- Helpers -----------------------------------------------------------------
def _highpass_filter_opm(raw):
    """ 'high-filter' = applies high-pass filter 3Hz ('Agressive'), low pass 40Hz, 60Hz notch """
    # apply high-pass filter 
    freq_min = 3
    freq_max = 40
    raw.load_data().filter(l_freq=freq_min, h_freq=freq_max)
    # always notch filter
    meg_picks = mne.pick_types(raw.info, meg=True)
    raw.notch_filter(freqs=60, picks=meg_picks)
    return raw
    
def _ssp_filter(raw):
    """ 'ssp-filter' = applies high-pass filter 2Hz, low pass 40Hz, 60Hz notch, and SSP method """
    # high pass
    freq_min = 2
    freq_max = 40
    raw.load_data().filter(l_freq=freq_min, h_freq=freq_max)
    # always notch filter
    meg_picks = mne.pick_types(raw.info, meg=True)
    raw.notch_filter(freqs=60, picks=meg_picks)
    # SSP projector
    proj = mne.compute_proj_raw(raw, start=0, stop=None, duration=1, n_grad=0, n_mag=1, n_eeg=0, reject=None, flat=None, n_jobs=None, meg='separate', verbose=None)
    raw_proj = raw.copy().add_proj(proj)
    return raw_proj

def _sss_prepros(raw):
    """ 'sss-filter' = applies high-pass filter 1Hz, low pass 40Hz, 60Hz notch, and SSS method """
    # freq_min = 1
    # freq_max = 40       
    # # # apply high-pass filter 
    # raw.load_data().filter(l_freq=freq_min, h_freq=freq_max)
    # # # apply notch filter for 60Hz power lines
    meg_picks = mne.pick_types(raw.info, meg=True)
    raw.notch_filter(freqs=60, picks=meg_picks)
    raw_sss = mne.preprocessing.maxwell_filter(raw, origin=(0., 0., 0.), int_order=8, ext_order=3, calibration=None, coord_frame='meg', regularize='in', ignore_ref=True, bad_condition='error', mag_scale=1.0, extended_proj=(), verbose=None)  
    freq_min = 2
    freq_max = 40       
    # # apply high-pass filter 
    raw_sss.load_data().filter(l_freq=freq_min, h_freq=freq_max)
    return raw_sss

def pros_OPM_data(raw, trigger_chan, prepros_type):
    """Load OPM raw data, find events on trigger chan, do specified preprocessing 

    Parameters
    ----------
    raw: mne.Raw
    trigger chan: str
        name of the trigger channel
    prepros_type: str
        pick from the following options 
        'high-filter' = applies high-pass filter 3Hz, low pass 40Hz, 60Hz notch filter.
        'ssp-filter' = applies high-pass filter 2Hz, low pass 40Hz, 60Hz notch, and SSP proj from baseline
        'sss-filter' = applies high-pass filter 1Hz, low pass 40Hz, 60Hz notch, and SSS method
        TODO: add Fosters Inverse with SSS
        
    Returns
    -------
    rawp : mne.Raw with applied filters/projectors
    events: arrary (event time x 3) containing event IDs and onsets
    """
    events = mne.find_events(raw, stim_channel=trigger_chan, shortest_event=1)
    ## preprocess
    if prepros_type == 'high-filter':
        rawp = _highpass_filter_opm(raw)
    elif prepros_type == 'ssp-filter':
        rawp = _ssp_filter(raw)
    elif prepros_type == 'sss-filter':
        rawp = _sss_prepros(raw)
    else:
        print("please pick preprocessing type from the defined options")
    return rawp,events


# --- Main (example usage) ---------------------------------------------------
if __name__ == '__main__':
    subjects_dir = '/Users/alexandria/Documents/STANFORD/FieldLine_tests/subjects/sub-XM'
    raw_files = ['20260206_143328_sub-XM_file-xantone_raw.fif']
    ## Define constants
    trigger_chan = 'di2' # should always be 'di2' for FieldLine but could be 'di1'
    
    for file in raw_files:
        if file.endswith(".fif"):
            """Do OPM-MEG load and preprocess """
            raw = mne.io.read_raw_fif(os.path.join(subjects_dir,file),'default', preload=True)
            ## Define filter type
            prepros_type = 'sss-filter' 
            [raw_pre, events] = pros_OPM_data(raw, trigger_chan, prepros_type)
            
            ## Get epochs and evoked response
            tmin = -0.2  # start of each epoch (200ms before the trigger)
            tmax = 0.6  # end of each epoch (600ms after the trigger)
            epochs = mne.Epochs(raw_pre, events, tmin=tmin, tmax=tmax, preload=True)
            evoked = epochs.average()
            fig = evoked.plot_joint()
            
        elif file.endswith(".ds"):
            """ TODO: add specific CTF preprocessing after we figure out event ID issues
            Do CTF-MEG load and preprocess """
            raw = read_raw_ctf(os.path.join(subjects_dir,file), preload=True)
        else:
            print("data file must be '.ds' for CTF or '.fif' for OPM MEG data")
    
        
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
    
    
    
    
    
    
