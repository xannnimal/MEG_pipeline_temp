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
# --- Load, find events, preprocess -------------------------------------------
def get_events_fif(raw,file):
    events = mne.find_events(raw, stim_channel=trigger_chan, shortest_event=1)
    if "VWFA" in file:
        task = "VWFA"
        event_id = { #VWFA
            "Cnd1": 1,
            "Cnd2": 2,
            "Cnd3": 3,
            "Cnd4": 4,
            "Cnd5": 5,
            "Cnd6": 6,
            "Cnd7": 7,
            "Cnd8": 8,
            "Cnd9": 9,
            "EndEpoch": 200
        }
    if "Tones" in file:
        task = "Tones"
        event_id = { #tones
            "Cnd11": 11,
            "Cnd12": 12,
            "Cnd13": 13,
            "Cnd14": 14,
            "Cnd15": 15,
            "EndEpoch": 200
        }
    if "V1Loc" in file:
        task = "V1Loc"
        event_id = { #V1Loc
            "Cnd16": 16,
            "EndEpoch": 200 
        }
    else:
        print("no valid events detected, please double check data file name")
    return events, event_id, task

def get_events_ctf(raw,file):
    events = mne.find_events(raw, stim_channel=trigger_chan, shortest_event=1)
    if "VWFA" in file:
        task= "VWFA"
        event_id = { #VWFA
            "Cnd1": 65536,
            "Cnd2": 131072,
            "Cnd3": 196608,
            "Cnd4": 262144,
            "Cnd5": 327680,
            "Cnd6": 393216,
            "Cnd7": 458752,
            "Cnd8": 524288,
            "Cnd9": 589824,
            "EndEpoch": 13107200 
        }
    if "Tones" in file:
        task = "Tones"
        event_id = { #tones
            "Cnd11": 720896,
            "Cnd12": 786432,
            "Cnd13": 851968,
            "Cnd14": 917504,
            "Cnd15": 983040,
            "EndEpoch": 13107200 
        }
    if "V1Loc" in file:
        task = "V1Loc"
        event_id = { #V1Loc
            "Cnd16": 1048576,
            "EndEpoch": 13107200 
        }
    else:
        print("no valid events detected, please double check data file name")
    return events, event_id, task

def filter_raw(raw):
    freq_min = 0.5
    freq_max = 80
    raw.load_data().filter(l_freq=freq_min, h_freq=freq_max)
    meg_picks = mne.pick_types(raw.info, meg=True)
    raw.notch_filter(freqs=60, picks=meg_picks)
    return raw
    
def ssp_filter(raw):
    # SSP projector
    proj = mne.compute_proj_raw(raw, start=0, stop=None, duration=1, n_grad=0, n_mag=1, n_eeg=0, reject=None, flat=None, n_jobs=None, meg='separate', verbose=None)
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
                 visualize=False, verbose=False):
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
    fwd_path = bem_path / f'{subject}-fwd.fif'

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
        mne.bem.make_watershed_bem(subject=subject, overwrite=True,
                                   volume='T1', atlas=True, gcaatlas=True,
                                   show=visualize)
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
        eeg=False, meg=True, mindist=mindist, n_jobs=None,
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
    from config_XM import sample_dir, raw_files, trans, subjects_dir, subject, viz_bool, save_report
    
    ## 1. Load and setup data
    for file in raw_files:
        # --- 1. Load data, find events ---------------------------------------
        if file.endswith(".fif"):
            ## load OPM, find events, do preprocessing
            ## specify trigger 
            trigger_chan = 'di2' # should always be 'di2' for FieldLine but could be 'di1'
            
            #setup raw, info, events, and specify task type
            raw = mne.io.read_raw_fif(os.path.join(sample_dir,file),'default', preload=True)
            [events,event_ids,task] = get_events_fif(raw,file)
            info = raw.info
            picks = 'mag'
            reject_criteria = dict(mag=4000e-15)  # 4000fT

            
        elif file.endswith(".ds"):
            raw = read_raw_ctf(os.path.join(sample_dir,file), preload=True)
            [events,event_ids,task] = get_events_ctf(raw,file)
            # always do this preprocessing, reccommended by Dylan @ UCSF
            raw.apply_gradient_compensation(3)
            info = raw.info
            picks = 'grad'
            reject_criteria = dict(grad=4000e-13) #4000 fT/cm
        else:
            print("data file must be '.ds' for CTF or '.fif' for OPM MEG data")
        
        # --- 1.B Look at Events -----------------------------------------------
        ## save events
        # mne.write_events( participant + '/' + participant + '_events.fif',events)
        if viz_bool:
            sfreq = raw.info["sfreq"]
            fig = mne.viz.plot_events(events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=event_ids)

        # --- 2. A Preprocess -------------------------------------------------
        ## start with methods common to both CTF and OPM-MEG
        #-- remove bad channels, check for NaNs
        bads = raw.info["bads"]
        raw.drop_channels(bads)
        bads_NaNs=[]
        for i in range(0,raw.info["nchan"]):
            ch_pos = raw.info["chs"][i]["loc"][:3]
            if np.isnan(ch_pos).any():
                bads_NaNs.append(raw.info["chs"][i]["ch_name"])
        raw.drop_channels(bads_NaNs)
        
        #-- Notch filter 60Hz, low pass 100Hz, High pass 0.5 Hz
        raw = filter_raw(raw)
        
        #-- Do SSS
        Lin=6
        raw_pre = sss_prepros(raw,Lin)
        
        #-- do SSP, one projector
        # raw_pre = ssp_filter(raw_sss)
        
        ## specific to task, device ??
        # reject eye blinks
        reject_blinks = False
        if reject_blinks ==True:
            eog_evoked = mne.preprocessing.create_eog_epochs(raw).average()
            eog_evoked.apply_baseline(baseline=(None, -0.2))
            eog_evoked.plot_joint()
            # eog_events = mne.preprocessing.find_eog_events(raw_pre)
            # onsets = eog_events[:, 0] / raw_pre.info["sfreq"] - 0.25
            # durations = [0.5] * len(eog_events)
            # descriptions = ["bad blink"] * len(eog_events)
            # blink_annot = mne.Annotations(
            #     onsets, durations, descriptions, orig_time=raw.info["meas_date"]
            # )
            # raw_pre.set_annotations(blink_annot)

        
        # --- 2.B visualize sensor alignment and BEM---------------------------------
        if viz_bool:
            mne.viz.plot_alignment(
                info,
                trans=trans,
                subject=subject,
                dig=True,
                meg=["helmet", "sensors"],
                subjects_dir=subjects_dir,
                surfaces="head-dense",
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
        tmin = -0.05  # start of each epoch (200ms before the trigger)
        tmax = 0.35  # end of each epoch (600ms after the trigger)
        # can add rejection criteria based on P-to-P signal        
        # separate out by event ID
        epochs = mne.Epochs(
            raw_pre, 
            events, 
            event_ids, 
            tmin, 
            tmax, 
            picks=picks,
            #reject=reject_criteria,
            #baseline=(None, 0)
            )
        evokeds = [epochs[name].average() for name in event_ids]
        conds = list(event_ids.keys())
        if viz_bool:
            for i in range(0,len(conds)):
                ts_args = ts_args = dict(time_unit="s") # can specify limits as ylim=dict(mag=(-400, 400)))
                topomap_args = dict(time_unit="s") # you can pass other args here, like 'vmin', 'vmax', 'cmap', etc.
                fig = evokeds[i].plot_joint(times="peaks", ts_args=ts_args, topomap_args=topomap_args,title=file + ' Task: '+ task + ', Condition: ' +conds[i] )
          
        # average over all conditions within the Task
        epochs_task = mne.Epochs(raw_pre, events, picks=[picks], tmin=tmin, tmax=tmax, preload=True)
        evoked = epochs_task.average()
        if viz_bool:
            ## specify plotting args
            ts_args = ts_args = dict(time_unit="s") 
            topomap_args = dict(time_unit="s") 
            fig = evoked.plot_joint(times="peaks", ts_args=ts_args, topomap_args=topomap_args, title= file+ ' Task: '+ task)
        
        # --- 4. Create covariance --------------------------------------------
        cov = mne.compute_covariance(epochs, tmax=0, projs=None, method="empirical", rank=None)
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
            os.path.join(sample_dir,file),
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
        inv_operator = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov, loose = 1, depth = None, fixed = False)
        # ## apply inverse for each condition 
                
        
        # # --- 7. Apply inverse to evokeds -------------------------------------
        method = "dSPM"  # could choose MNE, sLORETA, or eLORETA instead
        snr = 2.0
        lambda2 = 1.0 / snr**2
        stc, residual = apply_inverse(
            evoked,
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
            
        # evoked.crop(0.07, 0.13)
        # dip = mne.fit_dipole(evoked, cov, bem, trans)[0]
        # dip.plot_locations(trans, subject, subjects_dir)
       
        
        # # --- 8. Save all files in BIDS structure -----------------------------
        # #stc, inv_op = make_inverse(subjects_dir, subject_id, fwd, evoked, noise_cov)
        
        # --- 9. Generate and save MNE Report ---------------------------------
        if save_report:
            report = mne.Report(title="Report for subject: "+subject + ", Task: "+ task)
            report.add_raw(raw=raw, title= file , psd=True)
            report.add_trans(trans=trans, info=raw.info, title='Coregistration',subject=subject,subjects_dir=subjects_dir)
            report.add_events(events=events, title='Events from "events"', sfreq=sfreq)
            report.add_epochs(epochs=epochs, title='Epochs from "epochs"')
            report.add_evokeds(evokeds=evoked,titles= 'Evoked')
            report.add_covariance(cov=cov, info=raw.info, title="Covariance")
            report.add_bem(subject=subject, title='BEM')
            report.add_stc(stc=stc, title="STC")
            report.save(file + "report_raw.html", overwrite=True)
        



