#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 10:11:54 2026

@authors: Alexandria McPherson and Qiyuan Feng

C-SHARP MEG Forward & Inverse Pipeline
Supports both CTF (UCSF) and FieldLine OPM data.

**OPM-MEG: Do coregistration through YORC script FIRST to generate "trans" object
Full pipeline description: 
    1. Load data (CTF vs FIF files)
    2. Preprocessing - Reject eyeblinks with ICA, high-pass filter, add functions for SSS, homogenous field correction, etc
    3. Make evokeds for each condition
    4. Create covariance
    5. Forward solution
    6. Inverse solutions - explore options. Most take norm and throw away the orientation. Look into “free” orientation vs locked
    7. Apply inverse to evokeds
    8. Put into BIDS structure, specifically with events
    9. Generate MNE-report

"""
# --- Dependencies ------------------------------------------------------------
import os
import pandas as pd
import warnings
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt

import mne
from mne.io import read_raw_ctf
from mne import Covariance
from mne.minimum_norm import (make_inverse_operator, write_inverse_operator, apply_inverse)
from mne.beamformer import make_lcmv, apply_lcmv
from mne.surface import read_surface
# This takes care some numpy dependency issues...not required depending on the numpy version
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# --- FUNCTIONS ---------------------------------------------------------------
## Load data, set up events
#-----------------------------------------------------------
# -- events for CTF
def _get_correct_codes(events,special_codes):
    ## check for any accidental 192s, rename correct 200
    for i in range(0,np.shape(events)[0]):
        if events[i,1]==12582912:
            events[i-1,2]=13107200
    ## bit shift
    event_code_list = events[:, 2]
    for i in range(0,len(event_code_list)):
        event_code_list[i] = event_code_list[i] >> 16  
    events[:, 2]=event_code_list
    
    ## remove extra unexpected conditions
    deleted_Cnds = []
    new_events = []
    for i in range(0,len(event_code_list)):
        cnd = event_code_list[i]
        if events[i,1] == 0 and cnd in special_codes:
            new_events.append(events[i,:])
        else:
            deleted_Cnds.append(events[i,:])
    ## plot deleted conditions to inspect
    rejected_dict = _inspect_deleted_codes(deleted_Cnds)
    del rejected_dict
    event_code_list=np.array(new_events)[:,2]
    ## now things should look the same as FieldLine
    return np.array(new_events),np.array(event_code_list)

def _inspect_deleted_codes(deleted_Cnds):
    deleted = np.array(deleted_Cnds)
    deleted[:,1] = deleted[:,1] >> 16
    deleted = deleted[deleted[:,1] != 0]
    col_1 = np.array(deleted_Cnds)[:,0]
    col_3 = np.array(deleted_Cnds)[:,2]
    plt.figure()
    plt.scatter(deleted[:,0],deleted[:,1], color='blue', label='wrong triggers')
    plt.scatter(col_1,col_3, color='red', label='wrong sample occurance')
    plt.xlabel('Sample number of occurance')
    plt.ylabel('Registered Trigger Value (after bit shifting)')
    plt.title('Log of Deleted Triggers')
    plt.legend()
    plt.show()
    
    unique_values, counts = np.unique(col_3, return_counts=True)
    for i in range(0,len(unique_values)):
        print("Found "+ str(counts[i])+ " incorrect condition " +str(unique_values[i]))
    return dict(zip(unique_values, counts))


def get_events(raw,task,trigger_chan,modality):
    events = mne.find_events(raw, stim_channel=trigger_chan, shortest_event=1)
    if task =="VWFA":
        xDiva_codes = np.array([1,2,3,4,5,6,7,8,9,200])
        ## get cleaned events
        if modality=='CTF':
            [events,event_code_list] = _get_correct_codes(events,xDiva_codes)
        else:
            event_code_list = events[:, 2]
        event_code_updates = np.zeros_like(event_code_list)
        special_codes=np.array([1,2,3,4,5,6,7,8,9])
        ei = 0
        while ei < len(event_code_list):
            event = event_code_list[ei]
            if event in special_codes:
                event_code_updates[ei+1:ei+5] = event
                event_code_updates[ei] = 201  # code for what was the condition label
                ei += 4  # skip next 4 positions 
            else:
                ei += 1  # just advance by 1 if no match
        events[:, 2] = event_code_updates
        # get just the code part
        trigger_codes = events[:, 2]
        blocks = trigger_codes.reshape(-1, 5)
        blocks_rearranged = blocks[:, [1, 2, 3, 4, 0]]
        result = blocks_rearranged.flatten()
        events[:, 2] = result
        # make event codes interpretable
        code_dict = {1: 'highFreqWords_Sloan',
                     2: 'pseudowords_Sloan',
                     3: 'consonants_Sloan',
                     4: 'falseFontsHigh_Sloan',
                     5: 'highFreqWords_Courier',
                     6: 'pseudowords_Courier',
                     7: 'consonants_Courier',
                     8: 'falseFontsHigh_Courier',
                     9: 'background_',
                     201 : 'stimulusoffset_'
                     }
        # make into a nice pandas dataframe
        events_df = pd.DataFrame()
        events_df['code'] = events[:, 2]
        events_df['condition'] = [code_dict[c].split('_')[0] for c in events[:, 2]]
        events_df['font'] = [code_dict[c].split('_')[1] for c in events[:, 2]]
        
    if task=="Tones":
        special_codes = np.array([17,11,12,13,14,15,200])
        ## get cleaned events
        if modality=='CTF':
            [events,event_code_list] = _get_correct_codes(events,special_codes)
            code_dict = {17: '250_Hz', ##broken code
                         11: '500_Hz',
                         12: '1000_Hz',
                         13: '2000_Hz',
                         14: '4000_Hz',
                         15: 'background_',
                         200: 'stimulusoffset_'
                         }
        else:
            event_code_list = events[:, 2]
            code_dict = {10: '250_Hz', ##broken code
                         11: '500_Hz',
                         12: '1000_Hz',
                         13: '2000_Hz',
                         14: '4000_Hz',
                         15: 'background_',
                         200: 'stimulusoffset_'
                         }
        event_code_updates = np.zeros_like(event_code_list)
        # make into a nice pandas dataframe
        events_df = pd.DataFrame()
        events_df['code'] = events[:, 2]
        events_df['condition'] = [code_dict[c].split('_')[0] for c in events[:, 2]]
        events_df['units'] = [code_dict[c].split('_')[1] for c in events[:, 2]]
           
    if task == "V1Loc":
        if modality=='CTF':
            TRIAL_ID = 1048576   # trial-onset trigger (16 * 65536, equivalent to EEG DIN4)
            BIN_ID   = 13107200  # bin-onset trigger   (200 * 65536, equivalent to EEG DIN5)
        else:
            TRIAL_ID = 16   # trial-onset trigger (equivalent to EEG DIN4)
            BIN_ID   = 200  # bin-onset trigger   (equivalent to EEG DIN5)

        trial_samples = events[events[:, 2] == TRIAL_ID, 0]
        bin_samples   = events[events[:, 2] == BIN_ID,   0]

        event_id = {
            'bin/0': 0, 'bin/1': 1, 'bin/2': 2, 'bin/3': 3, 'bin/4': 4,
            'noise/prelude':  5,
            'noise/postlude': 6,
        }
        label_map = {v: k for k, v in event_id.items()}

        all_events = []
        for i, trial_start in enumerate(trial_samples):
            trial_end = trial_samples[i + 1] if i + 1 < len(trial_samples) else np.inf
            bins_in_trial = bin_samples[(bin_samples >= trial_start) & (bin_samples < trial_end)]

            for pos, s in enumerate(bins_in_trial[:5]):
                all_events.append([s, 0, pos])

            if len(bins_in_trial) >= 1:
                all_events.append([trial_start, 0, 5])   # prelude anchored at trial trigger

            if len(bins_in_trial) >= 6:
                all_events.append([bins_in_trial[5], 0, 6])  # postlude anchored at bin[5]

        all_events = np.array(all_events, dtype=int)
        all_events = all_events[np.argsort(all_events[:, 0])]
        events = all_events

        events_df = pd.DataFrame({'condition': [label_map[e[2]] for e in all_events]})

    else:
        print("no valid events detected, please double check data file name")
    return events_df, events
        
## -- Preprocessing Functions -------------------------------------------------
def filter_raw(raw,freq_min,freq_max):
    ## TODO
    raw.load_data().filter(l_freq=freq_min, h_freq=None)
    raw.filter(l_freq=None, h_freq=freq_max)
    meg_picks = mne.pick_types(raw.info, meg=True)
    raw.notch_filter(freqs=60, picks=meg_picks)
    return raw
    
def ssp_filter(raw):
    # SSP projector
    proj = mne.compute_proj_raw(raw, start=0, stop=None, duration=1, n_grad=0, n_mag=2, n_eeg=0, reject=None, flat=None, n_jobs=None, meg='separate', verbose=None)
    raw_proj = raw.copy().add_proj(proj)
    return raw_proj

def sss_prepros(raw,Lin):
    """do traditional SSS with origin 0 in MEG frame"""
    assert raw.info["bads"] == [] # double check bads were dropped
    raw_sss = mne.preprocessing.maxwell_filter(raw, origin=(0., 0., 0.), int_order=Lin, ext_order=3, calibration=None, coord_frame='meg', regularize='in', ignore_ref=True, bad_condition='error', mag_scale=100.0, extended_proj=(), verbose=None)  
    return raw_sss

def _eog_artifact(raw):
    """TODO: https://mne.tools/stable/auto_tutorials/preprocessing/20_rejecting_bad_data.html """
    eog_event_id = 512
    eog_events = mne.preprocessing.find_eog_events(raw, eog_event_id)
    raw.add_events(eog_events, "STI 014")
    return raw 


# --- Helpers -----------------------------------------------------------------

def _create_mid_surface(subjects_dir, subject, hemisphere):
    """Average white and pial surfaces to produce {hemisphere}.mid."""
    surf_dir = os.path.join(subjects_dir, subject, 'surf')

    white_coords, white_faces = nib.freesurfer.read_geometry(
        os.path.join(surf_dir, f'{hemisphere}.white'))
    pial_coords, pial_faces = nib.freesurfer.read_geometry(
        os.path.join(surf_dir, f'{hemisphere}.pial'))

    if not np.array_equal(white_faces, pial_faces):
        raise ValueError("Face mismatch between white and pial surfaces.")

    mid_coords = (white_coords + pial_coords) / 2.0
    mid_path = os.path.join(surf_dir, f'{hemisphere}.mid')
    nib.freesurfer.write_geometry(mid_path, mid_coords, white_faces)
    print(f"Mid surface saved: {mid_path}")


def _print_source_space_summary(subjects_dir, subject, src):
    """Print vertex counts for white, mid, and decimated source space."""
    surf_dir = os.path.join(subjects_dir, subject, 'surf')

    white_lh, _ = read_surface(os.path.join(surf_dir, 'lh.white'))
    white_rh, _ = read_surface(os.path.join(surf_dir, 'rh.white'))
    mid_lh, _   = read_surface(os.path.join(surf_dir, 'lh.mid'))
    mid_rh, _   = read_surface(os.path.join(surf_dir, 'rh.mid'))

    print(f"  White surface vertices : {white_lh.shape[0] + white_rh.shape[0]}")
    print(f"  Mid surface vertices   : {mid_lh.shape[0] + mid_rh.shape[0]}")
    print(f"  Decimated src vertices : {sum(len(h['vertno']) for h in src)}")


def _visualize_source_space(src, subject, surface):
    """Plot BEM cross-section and 3-D source alignment.

    NOTE: 3-D plots (PyVista/Mayavi) are non-blocking by default in recent
    MNE versions, so they should not stall the pipeline. If running in a
    headless environment set ``mne.viz.set_3d_backend('notebook')`` or skip
    visualization entirely.
    """
    mne.viz.plot_bem(src=src, subject=subject,
                     brain_surfaces=surface, orientation='coronal')
    fig = mne.viz.plot_alignment(
        subject=subject, surfaces="white",
        coord_frame="mri", src=src,
    )
    mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75,
                        distance=0.30, focalpoint=(-0.03, -0.01, 0.03))


def _visualize_forward(evoked, trans, fwd, subject):
    """Plot sensor-source alignment with forward solution overlay."""
    mne.viz.plot_alignment(
        evoked.info, trans=trans, fwd=fwd,
        subject=subject, surfaces="white",
    )

# --- Covariance Matrix --------------------------------------------------------

# currently, make_inverse is not calling make_cov and takes noise_cov as an input directly
# because this function needs epochs instead of evoked data
def make_cov(epochs, identity=False):
    # simply compute covariance, can be changed later
    noise_cov = mne.compute_covariance(epochs)

    if identity:
        # make the noise covariance identity if:
        # 1) don't have a good estimate of the noise covariance, or
        # 2) the baseline produces unstable results, or
        # 3) for noise-free simulation testing
        n=len(noise_cov.data) 
        for j in range(n):
            for k in range(n):
                if (j==k):
                    noise_cov.data[j][k]=1
                else:
                    noise_cov.data[j][k]=0

def _get_identity_cov(fwd, evoked):
    """Identity noise covariance (for noise-free / simulation testing)."""
    return Covariance(
        data=np.eye(fwd["sol"]["data"].shape[0]),
        names=evoked.info['ch_names'],
        bads=[],
        nfree=1,
        projs=[],
    )


# --- Forward Solution --------------------------------------------------------

def make_forward(subject_id, subjects_dir, trans, evoked,
                 overwrite_fwd=True, overwrite=False,
                 fixed=True, bem_ico=4, src_space="oct7",
                 conductivity=(0.3, 0.006, 0.3),
                 mindist=5, surface='mid',
                 visualize=False, verbose=False, eeg=False, meg=True):
    """Build (or load) the forward solution for a subject.

    Parameters
    ----------
    subject_id : str
        FreeSurfer subject name.
    trans : str or mne.Transform
        Head-to-MRI transform (-trans.fif path or Transform object).
    evoked : mne.Evoked
        Used for sensor info when computing the leadfield.
    subjects_dir : str or Path
        FreeSurfer SUBJECTS_DIR.
    overwrite_fwd : bool
        If False, read existing forward if available.
    overwrite : bool
        If True, recompute BEM and source space even if files exist.
    fixed : bool
        Convert to fixed (surface-normal) orientation.
    bem_ico : int
        ICO decimation level for BEM surfaces (higher = finer mesh).
    src_space : str
        Source space decimation, e.g. 'oct6', 'oct7'.
    conductivity : tuple
        BEM conductivities in S/m.
        MEG-only: (0.3,) — single shell.
        EEG+MEG:  (0.3, 0.006, 0.3) — scalp / skull / brain.
    mindist : float
        Exclude sources closer than this (mm) to the inner skull.
    surface : str
        Source surface type ('white', 'pial', or 'mid').
    visualize : bool
        Show BEM / alignment plots (requires a display backend).
    verbose : bool
        Print extra source-space and leadfield diagnostics.

    Returns
    -------
    fwd : mne.Forward
    """
    subject = subject_id
    subjects_dir = str(subjects_dir)
    bem_path = Path(subjects_dir) / subject / 'bem'
    modality_tag = 'eeg' if (eeg and not meg) else ('meg' if (meg and not eeg) else 'meeg')
    fwd_path = bem_path / f'{subject}-{modality_tag}-fwd.fif'

    # --- Try loading existing forward ---
    if fwd_path.exists() and not overwrite_fwd:
        print("---- Reading existing forward solution ----")
        fwd = mne.read_forward_solution(str(fwd_path))
        if fixed:
            fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)
        leadfield = fwd["sol"]["data"]
        print(f"Leadfield: {leadfield.shape[0]} sensors x {leadfield.shape[1]} dipoles")
        return fwd

    # --- Build from scratch ---
    print("---- Building forward solution ----")

    # Mid surface generation
    if surface == 'mid':
        for hemi in ['lh', 'rh']:
            _create_mid_surface(subjects_dir, subject, hemi)

    # --- BEM model & solution ---
    if isinstance(conductivity, tuple) and len(conductivity) == 3:
        scale = str(round(1 / (conductivity[1] / conductivity[2])))
    else:
        scale = "1layer"

    bem_fname = bem_path / f'{subject}-scale{scale}-ico{bem_ico}-bem.fif'
    bem_sol_fname = bem_path / f'{subject}-scale{scale}-ico{bem_ico}-bem-sol.fif'

    if not bem_fname.exists() or overwrite:
        mne.bem.make_watershed_bem(subject=subject, overwrite=True,volume='T1',atlas=True, gcaatlas=True,show=visualize)              
        model = mne.make_bem_model(subject=subject, ico=bem_ico,
                                   conductivity=conductivity)
        mne.write_bem_surfaces(str(bem_fname), model, overwrite=True)
        bem = mne.make_bem_solution(model)
        mne.write_bem_solution(str(bem_sol_fname), bem, overwrite=True)
    else:
        print("---- Reading existing BEM ----")
        bem = mne.read_bem_solution(str(bem_sol_fname))

    # --- Source space ---
    src_fname = bem_path / f'{subject}-{src_space}-src.fif'

    if not src_fname.exists() or overwrite:
        src = mne.setup_source_space(subject=subject, spacing=src_space,
                                     surface=surface, add_dist=False)
        mne.write_source_spaces(str(src_fname), src, overwrite=True)
        if visualize:
            _visualize_source_space(src, subject, surface)
        if verbose:
            _print_source_space_summary(subjects_dir, subject, src)
    else:
        print("---- Reading existing source space ----")
        src = mne.read_source_spaces(str(src_fname))

    # --- Compute forward ---
    fwd = mne.make_forward_solution(
        evoked.info, trans=trans, src=src, bem=bem,
        eeg=eeg, meg=meg, mindist=mindist, n_jobs=None,
    )
    mne.write_forward_solution(str(fwd_path), fwd, overwrite=True)

    if fixed:
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)

    if verbose:
        leadfield = fwd["sol"]["data"]
        print(f"Source space before fwd: {src}")
        print(f"Source space after fwd:  {fwd['src']}")
        print(f"Leadfield: {leadfield.shape[0]} sensors x {leadfield.shape[1]} dipoles")
        print(f"Dipole orientations shape: {fwd['source_nn'].shape}")

    if visualize:
        _visualize_forward(evoked, trans, fwd, subject)

    return fwd



# --- Inverse Solution --------------------------------------------------------

def make_inverse(subjects_dir, subject, fwd, evoked, noise_cov,
                 fixed_ori=True, noise_free=False,
                 snr=8, lambda2=None,
                 inverse_method="MNE",
                 save_dir=None,
                 loose=0.2, depth=0.8):
    """Compute the inverse operator and apply it to an evoked.

    Parameters
    ----------
    subjects_dir : str or Path
    subject : str
    fwd : mne.Forward
    evoked : mne.Evoked
    noise_cov : mne.Covariance
        Ignored when noise_free=True (identity cov is used instead).
    fixed_ori : bool
        True  -> fixed orientation (loose=0, depth=None).
        False -> free/loose orientation using `loose` and `depth` params.
    noise_free : bool
        Use identity covariance and high SNR (simulation mode).
    snr : float
        Signal-to-noise ratio; lambda2 = 1/snr^2 when lambda2 is None.
    lambda2 : float or None
        Explicit regularisation. Overrides snr if provided.
    inverse_method : str
        'MNE', 'dSPM', 'sLORETA', 'eLORETA', 'beamformer', or 'customized'.
    save_dir : str or Path, optional
        If provided, save the resulting STC here.
    loose : float
        Loose orientation constraint (only used when fixed_ori=False).
    depth : float
        Depth weighting exponent (only used when fixed_ori=False).

    Returns
    -------
    stc : mne.SourceEstimate
    inverse_operator : InverseOperator
    """
    print("---- Computing inverse ----")

    # Noise covariance
    if noise_free:
        noise_cov = _get_identity_cov(fwd, evoked)
        snr = 100
    common_chs = [ch for ch in evoked.ch_names if ch in noise_cov.ch_names]
    noise_cov = noise_cov.pick_channels(common_chs)

    # Regularisation
    if lambda2 is None:
        lambda2 = 1.0 / snr ** 2
    print(f"  SNR={snr}, lambda2={lambda2:.4f}")

    # Inverse operator
    if fixed_ori:
        inverse_operator = make_inverse_operator(
            evoked.info, fwd, noise_cov, loose=0, depth=None, fixed=True)
    else:
        inverse_operator = make_inverse_operator(
            evoked.info, fwd, noise_cov, loose=loose, depth=depth, fixed=False)

    inv_fname = Path(subjects_dir) / subject / 'bem' / f'{subject}-inv.fif'
    write_inverse_operator(str(inv_fname), inverse_operator, overwrite=True)

    # Apply inverse
    if inverse_method == "beamformer":
        filters = make_lcmv(
            info=evoked.info, forward=fwd, data_cov=noise_cov,
            reg=0.05,
            pick_ori=None if fixed_ori else 'normal',
            rank='info', verbose=False,
        )
        stc = apply_lcmv(evoked, filters)

    elif inverse_method == "customized":
        stc = _pseudo_inverse_custom(subject, fwd, evoked, snr,
                                     fixed_ori=fixed_ori)
    else:
        # MNE, dSPM, sLORETA, eLORETA
        stc = apply_inverse(
            evoked, inverse_operator, lambda2,
            method=inverse_method, pick_ori=None, verbose=True,
        )

    # Save STC if output directory was given
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stc_fname = str(save_dir / f'{subject}-{inverse_method}')
        stc.save(stc_fname, overwrite=True)  # mne.SourceEstimate.save()
        print(f"  STC saved: {stc_fname}")

    return stc, inverse_operator


# --- Custom SVD pseudo-inverse ----------------------------------------------

def _pseudo_inverse_custom(subject, fwd, evoked, snr, fixed_ori):
    """Truncated SVD pseudo-inverse (for experimentation / comparison).

    Computes A_pinv via SVD of the leadfield, thresholding small singular
    values based on the SNR.  For free orientation, the 3-component source
    vector is collapsed to its norm.
    """
    A = fwd["sol"]["data"]  # sensors x dipoles
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)

    threshold = np.max(Sigma) / (snr + 1)
    Sigma_inv = np.array([1/s if s > threshold else 0 for s in Sigma])
    n_kept = int(np.sum(Sigma > threshold))

    print("---- Custom pseudo-inverse (SVD) ----")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Kept {n_kept}/{len(Sigma)} singular values")

    A_pinv = Vt.T @ np.diag(Sigma_inv) @ U.T
    X = A_pinv @ evoked.data  # estimated source activity

    # Collapse xyz triplets to vector norm for free orientation
    if not fixed_ori:
        n_sources = X.shape[0] // 3
        X = np.linalg.norm(X.reshape(n_sources, 3, -1), axis=1)

    print(f"  Source estimate shape: {X.shape}")

    vertices = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
    stc = mne.SourceEstimate(
        data=X, vertices=vertices,
        tmin=evoked.times[0],
        tstep=1.0 / evoked.info['sfreq'],
        subject=subject,
    )
    return stc

# ---- Helpers from Teresa Cheung --------------------------------------------
def plot3Dhelmetwithhpi(raw,ax,showLabels=True,showDevice=True,thetitle=''):

    raw.load_data()

    ax.title.set_text(thetitle)
    trans=raw.info['dev_head_t']['trans']
    print(trans)

    info=raw.info
    digpts=np.array([],dtype=float)
    digpts_head=np.array([],dtype=float)

    offset=0
    picks = mne.pick_types(info, meg='mag')
    for j in picks:
        ch = info['chs'][j]
        #print(ch['ch_name'])
        #print(ch['loc'][0:3])
        ex=ch['loc'][3:6]
        ey=ch['loc'][6:9]
        ez=ch['loc'][9:12]
        R=np.vstack((ex, ey, ez))
        #take the loc points and offset by 5 mm to account for distance from scalp to sensor
        move = np.dot((0,0,offset),R)
        digpts=np.append(digpts,(ch['loc'][0:3]-move)) # to account for the gap between sensor surface and cell centre
        head_coord = mne.transforms.apply_trans(trans, ch['loc'][0:3]-move)
        digpts_head=np.append(digpts_head,(head_coord)) # to account for the gap between sensor surface and cell centre
        #digpts=np.append(digpts,ch['loc'][0:3])
        #print((ch['loc'][0:3]),(ch['loc'][0:3]-move))
        #print(move)

    n=int(digpts.shape[0]/3)
    digpts=digpts.reshape((n,3))

    n=int(digpts_head.shape[0]/3)
    digpts_head=digpts_head.reshape((n,3))

    print(digpts.shape)

    

    thesize=7

    if showDevice:
        for i in range(len(digpts)):
            ax.scatter(digpts[i][0], digpts[i][1], digpts[i][2], '10', c='cyan', alpha=0.75)
        i=0
        ax.scatter(digpts[i][0], digpts[i][1], digpts[i][2], '10', c='cyan',s =40, alpha=0.75)
        
        ax.text(digpts[i][0], digpts[i][1], digpts[i][2],  raw.info['ch_names'][i], size=thesize, zorder=1,  color='cyan') 

    if 1:
        for i in range(len(digpts_head)):
            ax.scatter(digpts_head[i][0], digpts_head[i][1], digpts_head[i][2], '10', c='magenta', alpha=0.75)
            if showLabels:
                ax.text(digpts_head[i][0], digpts_head[i][1], digpts_head[i][2],  raw.info['ch_names'][i], size=thesize, zorder=1,  color='magenta') 

        i=0
    
        ax.scatter(digpts_head[i][0], digpts_head[i][1], digpts_head[i][2], '10', c='magenta',s=40, alpha=0.75)
        ax.text(digpts_head[i][0], digpts_head[i][1], digpts_head[i][2],  raw.info['ch_names'][i], size=thesize, zorder=1,  color='magenta') 

    LPA=raw.info['dig'][4]['r']
    NA=raw.info['dig'][3]['r']
    RPA=raw.info['dig'][5]['r']
    IN=raw.info['dig'][6]['r']
    CZ=raw.info['dig'][7]['r']

    ax.scatter(NA[0],NA[1],NA[2], '10', s=80, c='blue', marker='^', alpha=0.75)
    ax.text(NA[0],NA[1],NA[2],  'NA', size=thesize, zorder=1,  color='blue') 

    ax.scatter(LPA[0],LPA[1],LPA[2], '10', s=80, c='green', marker='^',alpha=0.75)
    ax.text(LPA[0],LPA[1],LPA[2],   'LPA', size=thesize, zorder=1,  color='green') 

    ax.scatter(RPA[0],RPA[1],RPA[2], '10', s=80, c='red', marker='^',alpha=0.75)
    ax.text(RPA[0],RPA[1],RPA[2],   'RPA', size=thesize, zorder=1,  color='red') 

    ax.scatter(IN[0],IN[1],IN[2], '10', s=80, c='black', marker='^',alpha=0.75)
    ax.scatter(CZ[0],CZ[1],CZ[2], '10', s=80, c='black', marker='^',alpha=0.75)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


# -----------------------------------------------------------------------------
# --- Main (example usage) ----------------------------------------------------
if __name__ == '__main__':
   # --- Load user-specific config (copy config_template.py -> config.py and fill in your paths)
    from config_XM import sample_dir, raw_files, trans, subjects_dir, subject, task, modality, viz_bool, sss_bool, save_report, report_dir, save_raw
    ## if getting a FreeSurfer error, set this variable to the location of your subjects anatomy
    # os.environ["SUBJECTS_DIR"] = subjects_dir
    

    ## 1. Load and setup data
    for file in raw_files:
        # --- 1. Load data, find events ---------------------------------------
        # if np.size(file) == 1 and modality=='OPM':
        #     ## load OPM, find events, do preprocessing
        #     ## specify trigger 
        #     trigger_chan = 'di2' # should always be 'di2' for FieldLine but could be 'di1'
            
        #     #setup raw, info, events, and specify task type
        #     raw = mne.io.read_raw_fif(os.path.join(sample_dir,file),'default', preload=True)
        #     [events_df,events] = get_events_fif(raw,task,trigger_chan)
        #     info = raw.info
        #     picks = 'mag'
        #     reject_criteria = dict(mag=4000e-15)  # 4000fT
            
        ## -- get Raw, check for datasets that need concatenating --------------------------
        if task == 'VWFA' and modality=='CTF' and (subject=='S001' or subject=='S009'):
            trigger_chan='STIM'
            raw = read_raw_ctf(os.path.join(sample_dir,file[0]), preload=True)
            mne.io.concatenate_raws([raw,read_raw_ctf(os.path.join(sample_dir,file[1]), preload=True)], on_mismatch="ignore")
            #[events_df,events] = get_events_ctf(raw,task,trigger_chan)
            # always do this preprocessing, reccommended by Dylan @ UCSF
            raw.apply_gradient_compensation(3)
            info = raw.info
            picks = 'grad'
            reject_criteria = dict(grad=4000e-13) #4000 fT/cm
                
        elif modality=='CTF':  
            trigger_chan='STIM'
            raw = read_raw_ctf(os.path.join(sample_dir,file), preload=True)
            #[events_df,events] = get_events_ctf(raw,task,trigger_chan)
            # always do this preprocessing, reccommended by Dylan @ UCSF
            raw.apply_gradient_compensation(3)
            info = raw.info
            picks = 'grad'
            reject_criteria = dict(grad=4000e-13) #4000 fT/cm
            
        elif modality == 'OPM':
            trigger_chan = 'di2'
            raw = mne.io.read_raw_fif(os.path.join(sample_dir,file),'default', preload=True)
            #[events_df,events] = get_events_fif(raw,task,trigger_chan)
            info = raw.info
            picks = 'mag'
            reject_criteria = dict(mag=4000e-15)  # 4000fT
        else:
            print("data file must be '.ds' for CTF or '.fif' for OPM MEG data")
        
        ## -- Load and Setup Events
        [events_df,events] = get_events(raw,task,trigger_chan,modality)
        
        # --- 1.B Look at Events -----------------------------------------------
        ## save events
        # mne.write_events( participant + '/' + participant + '_events.fif',events)
        sfreq = raw.info["sfreq"]
        if viz_bool:
            fig = mne.viz.plot_events(events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp)
        ##set up report
        if save_report:
            report = mne.Report(title="Report for subject: "+subject + ", Task: "+ task)
            report.add_raw(raw=raw, title= subject +', '+modality+', '+task, psd=True)
            report.add_trans(trans=trans, info=raw.info, title='Coregistration',subject=subject,subjects_dir=subjects_dir)
            report.add_events(events=events, title='Events from "events"', sfreq=sfreq)
            

        # --- 2. A Preprocess -------------------------------------------------
        ## start with methods common to both CTF and OPM-MEG
        #-- remove bad channels, check for NaNs
        bads = raw.info["bads"]
        bads_NaNs=[]
        for i in range(0,raw.info["nchan"]):
            ch_pos = raw.info["chs"][i]["loc"][:3]
            if np.isnan(ch_pos).any():
                bads_NaNs.append(raw.info["chs"][i]["ch_name"])
        raw.info["bads"].extend(bads_NaNs)
        raw.drop_channels(bads)
        ## TODO: add another step for looking at bad channels, when to mark bad on OPM -- obvious in FFT
        
        #-- Notch filter 60Hz, low pass and high pass
        ## TODO split freq by task
        freq_min = 0.5
        freq_max = 80
        raw = filter_raw(raw,freq_min,freq_max)
        #downsample?
        
        #-- Do SSS
        if sss_bool:
            Lin=8
            raw = sss_prepros(raw,Lin)
        
        #-- do SSP, one projector
        #raw_pre = ssp_filter(raw)
        
        if save_raw:
            raw_clean_path = os.path.join(sample_dir,f'sub-{subject}_task-{task}_preproc_raw.fif')
            raw.save(raw_clean_path, overwrite=True)
            print(f"Clean raw saved → {raw_clean_path}")
        
        # --- 2.B visualize sensor alignment and BEM---------------------------------
        if viz_bool:
            mne.viz.plot_alignment(
                info,
                trans=trans,
                subject=subject,
                dig=True,
                meg=["helmet", "sensors"],
                subjects_dir=subjects_dir,
                surfaces="head",
                )
            # look at BEM
            plot_bem_kwargs = dict(
                subject=subject,
                subjects_dir=subjects_dir,
                brain_surfaces="white",
                orientation="coronal",
                slices=[50, 100, 150, 200])
            mne.viz.plot_bem(**plot_bem_kwargs)
            
        # --- 3. Make Epochs and Evokeds --------------------------------------
        # Task-specific epoch window
        if task == "V1Loc":
            tmin = 0.0
            tmax = 2.0
            sfreq = raw.info['sfreq']
            n_samples = int(round((tmax - tmin) * sfreq))
            tmax = tmin + (n_samples - 1) / sfreq  # exact sample-aligned tmax
        elif task == "VWFA":
            tmin = 0.0
            tmax = 1.0
        else:
            ## TODO: check Tones vs VWFA task specific epoch timing
            tmin = -0.05
            tmax = 0.3

        # Single shared Epochs call — all tasks use metadata for condition labels
        epochs = mne.Epochs(raw, events,
                    tmin=tmin, tmax=tmax,
                    baseline=None,
                    reject=None,
                    preload=True, metadata=events_df)

        # Build evokeds — V1Loc groups bin/* and noise/* together; other tasks average per condition
        if task == "V1Loc":
            evokeds = {
                'bin':   epochs[epochs.metadata['condition'].str.startswith('bin/')].average(),
                'noise': epochs[epochs.metadata['condition'].str.startswith('noise/')].average(),
            }
        else:
            evokeds = dict()
            query = "condition == '{}'"
            for cond in epochs.metadata['condition'].unique():
                evokeds[str(cond)] = epochs[query.format(cond)].average()

        # Shared report and visualization
        for cond, ev in evokeds.items():
            if save_report:
                report.add_evokeds(evokeds=ev, titles=[cond])
            if viz_bool:
                ts_args = dict(time_unit="s")
                topomap_args = dict(time_unit="s")
                ev.plot_joint(times="peaks", ts_args=ts_args, topomap_args=topomap_args,
                              title=subject + ' Task: ' + task + ', Condition: ' + cond)
        
        # evokeds = [epochs[name].average() for name in event_ids]
        # conds = list(event_ids.keys())
          
        # # average over all conditions within the Task
        # epochs_task = mne.Epochs(raw_pre, events, picks=[picks], tmin=tmin, tmax=tmax, preload=True)
        # evoked = epochs_task.average()
        # if viz_bool:
        #     ## specify plotting args
        #     ts_args = ts_args = dict(time_unit="s") 
        #     topomap_args = dict(time_unit="s") 
        #     fig = evoked.plot_joint(times="peaks", ts_args=ts_args, topomap_args=topomap_args, title= file+ ' Task: '+ task)
        
        if save_raw:
            epochs_fif_path = os.path.join(sample_dir,f'sub-{subject}_task-{task}_preproc_epo.fif')
            epochs.save(epochs_fif_path, overwrite=True)
            print(f"Clean epochs saved → {epochs_fif_path}")

        # --- 4. Create covariance --------------------------------------------
        cov = mne.compute_covariance(epochs, tmax=0, projs=None, method="empirical", rank='info')
        # cov.save(f'{sample_dir}/V1Loc_empirical_cov.fif')
        # cov = mne.cov.regularize(cov, evoked.info, mag=0.05, grad = 0.05, proj = True, exclude = 'bads')


        # --- 5. Forward solution ---------------------------------------------
        # ##fwd = make_forward(subject, trans, evoked, subjects_dir)
        src = mne.setup_source_space(subject, spacing="ico4", add_dist="patch", subjects_dir=subjects_dir)
        # conductivity = (0.3,)  # for single layer
        conductivity = (0.3, 0.006, 0.3)  # for three layers
        model = mne.make_bem_model(subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        fwd = mne.make_forward_solution(
            raw.info,
            trans=trans,
            src=src,
            bem=bem,
            meg=True,
            eeg=False,
            mindist=5.0,
            n_jobs=None,
            verbose=True,
        )
        # src = mne.setup_source_space(subject, spacing="ico4", add_dist="patch", subjects_dir=subjects_dir)
        # conductivity = (0.3,)  # for single layer
        # # conductivity = (0.3, 0.006, 0.3)  # for three layers
        # model = mne.make_bem_model(subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir)
        # bem = mne.make_bem_solution(model)
        # fwd = mne.make_forward_solution(
        #     os.path.join(sample_dir,file),
        #     trans=trans,
        #     src=src,
        #     bem=bem,
        #     meg=True,
        #     eeg=False,
        #     mindist=5.0,
        #     n_jobs=None,
        #     verbose=True,
        # )
        # This version gave error "Surface inner skull is not completely inside surface outer skull"?
        # update 2026 Apr 7 (QF): make_forward finished without error
        # fwd = make_forward(subject, subjects_dir, trans, evoked,
        #                  overwrite_fwd=False, overwrite=False,
        #                  fixed=True, bem_ico=4, src_space="oct7",
        #                  conductivity=(0.3, 0.006, 0.3),
        #                  mindist=5, surface='mid',
        #                  visualize=False, verbose=False)
        
        # --- 6. Inverse solution ---------------------------------------------
        # fwd = mne.read_forward_solution(f'{subjects_dir}/{subject}/bem/{subject}-fwd.fif')
        # cov = mne.read_cov(f'{sample_dir}/V1Loc_empirical_cov.fif')
        # stc, inv_op = make_inverse(subjects_dir, subject, fwd, evoked, cov, inverse_method="dSPM")
        for cond in evokeds.keys():
            # fwd = make_forward(subject, subjects_dir, trans, evokeds[cond],
            #                  overwrite_fwd=False, overwrite=False,
            #                  fixed=True, bem_ico=4, src_space="oct7",
            #                  conductivity=(0.3, 0.006, 0.3),
            #                  mindist=5, surface='mid',
            #                  visualize=False, verbose=False)
            inv_operator = mne.minimum_norm.make_inverse_operator(evokeds[cond].info, fwd, cov, loose = 1, depth = None, fixed = False)
            # # --- 7. Apply inverse to evokeds -------------------------------------
            method = "dSPM"  # could choose MNE, sLORETA, or eLORETA instead
            snr = 2.0
            lambda2 = 1.0 / snr**2
            stc, residual = apply_inverse(
                evokeds[cond],
                inv_operator,
                lambda2,
                method=method,
                pick_ori=None,
                return_residual=True,
                verbose=True)
            
            # --- 8. Visualize inverse --------------------------------------------
            if viz_bool: 
                vertno_max, time_max = stc.get_peak(hemi="rh")
                surfer_kwargs = dict(
                    hemi="split",
                    subjects_dir=subjects_dir, # clim=dict(kind="value"), lims=[12,20,28]
                    views=["caudal", "medial"], # for visual task
                    initial_time=time_max,
                    time_unit="s",
                    size=(800, 800),
                    smoothing_steps=5)

                brain = stc.plot(**surfer_kwargs)
                brain.plotter.scalar_bar.GetLabelTextProperty().SetFontSize(8)
                # These params are tested to fit if you have two views (one on top and one at the bottom) with splitted hemi. eg. views=["caudal", "medial"]
                # You should change this if you have single view only
                sb = brain.plotter.scalar_bar
                x, _ = sb.GetPosition()
                sb.SetPosition(x, 0.6)  # vertically between caudal (top) and medial (bottom) rows
                w, h = sb.GetPosition2()
                sb.SetPosition2(w, h * 0.5)  # shrink height so top of the color bar doesn't clip
                for renderer in brain.plotter.renderers:
                    for actor in renderer.GetActors2D():
                        if hasattr(actor, 'GetTextProperty'):
                            actor.GetTextProperty().SetFontSize(7)
                brain.add_foci(
                    vertno_max,
                    coords_as_verts=True,
                    hemi="rh",
                    color="blue",
                    scale_factor=0.6,
                    alpha=0.5)
                brain.add_text(0.1, 0.9, "dSPM, Task: " + task, "title", font_size=8)
                
        # # --- 8. Save all files in BIDS structure -----------------------------
        # #stc, inv_op = make_inverse(subjects_dir, subject_id, fwd, evoked, noise_cov)
        
        # --- 9. Generate and save MNE Report ---------------------------------
        if save_report:
            report.add_epochs(epochs=epochs, title='Epochs from "epochs"')
            report.add_covariance(cov=cov, info=raw.info, title="Covariance")
            report.add_bem(subject=subject, title='BEM')
            report.add_stc(stc=stc, title="STC")
            report_dir=report_dir
            report.save(report_dir+ subject + task + "report_raw.html", overwrite=True)
