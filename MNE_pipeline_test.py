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

import mne
from mne.io import read_raw_ctf
from mne.minimum_norm import apply_inverse
from mne import Covariance
from mne.minimum_norm import (make_inverse_operator, write_inverse_operator, apply_inverse)
from mne.beamformer import make_lcmv, apply_lcmv
from mne.surface import read_surface
# This takes care some numpy dependency issues...not required depending on the numpy version
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# --- FUNCTIONS ---------------------------------------------------------------
# --- Load, find events, preprocess -------------------------------------------
def _highpass_filter_opm(raw):
    """ 'high-filter' = applies high-pass filter 3Hz ('Agressive'), low pass 40Hz, 60Hz notch """
    freq_min = 3
    freq_max = 40
    raw.load_data().filter(l_freq=freq_min, h_freq=freq_max)
    meg_picks = mne.pick_types(raw.info, meg=True)
    raw.notch_filter(freqs=60, picks=meg_picks)
    return raw
    
def _ssp_filter(raw):
    """ 'ssp-filter' = applies high-pass filter 2Hz, low pass 40Hz, 60Hz notch, and SSP method """
    freq_min = 2
    freq_max = 40
    raw.load_data().filter(l_freq=freq_min, h_freq=freq_max)
    meg_picks = mne.pick_types(raw.info, meg=True)
    raw.notch_filter(freqs=60, picks=meg_picks)
    # SSP projector
    proj = mne.compute_proj_raw(raw, start=0, stop=None, duration=1, n_grad=0, n_mag=1, n_eeg=0, reject=None, flat=None, n_jobs=None, meg='separate', verbose=None)
    raw_proj = raw.copy().add_proj(proj)
    return raw_proj

def _sss_prepros(raw):
    """ 'sss-filter' = applies high-pass filter 1Hz, low pass 40Hz, 60Hz notch, and SSS method """
    meg_picks = mne.pick_types(raw.info, meg=True)
    raw.notch_filter(freqs=60, picks=meg_picks)
    raw_sss = mne.preprocessing.maxwell_filter(raw, origin=(0., 0., 0.), int_order=8, ext_order=3, calibration=None, coord_frame='meg', regularize='in', ignore_ref=True, bad_condition='error', mag_scale=1.0, extended_proj=(), verbose=None)  
    freq_min = 2
    freq_max = 40       
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
    noise_cov = noise_cov.pick_channels(evoked.ch_names)

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


# -----------------------------------------------------------------------------
# --- Main (example usage) ----------------------------------------------------
if __name__ == '__main__':
    # subjects_dir = Path(os.environ["SUBJECTS_DIR"])
    subjects_dir = '/Users/alexandria/Documents/STANFORD/FieldLine_tests/subjects/sub-XM'
    raw_files = ['20260206_143328_sub-XM_file-xantone_raw.fif']
    ## Define constants
    trigger_chan = 'di2' # should always be 'di2' for FieldLine but could be 'di1'
    
    for file in raw_files:
        # --- 1. and 2. Load data, find events, and preprocess ----------------
        if file.endswith(".fif"):
            ## load OPM, find events, do preprocessing
            raw = mne.io.read_raw_fif(os.path.join(subjects_dir,file),'default', preload=True)
            ## Define filter type
            prepros_type = 'sss-filter' 
            [raw_pre, events] = pros_OPM_data(raw, trigger_chan, prepros_type)
            
        elif file.endswith(".ds"):
            """ TODO: add specific CTF preprocessing after we figure out event ID issues
            Do CTF-MEG load and preprocess """
            raw = read_raw_ctf(os.path.join(subjects_dir,file), preload=True)
        else:
            print("data file must be '.ds' for CTF or '.fif' for OPM MEG data")
        
        # --- 3. Make Epochs and Evokeds --------------------------------------
        """TODO: do this for each event/condition type """
        tmin = -0.2  # start of each epoch (200ms before the trigger)
        tmax = 0.6  # end of each epoch (600ms after the trigger)
        epochs = mne.Epochs(raw_pre, events, tmin=tmin, tmax=tmax, preload=True)
        evoked = epochs.average()
        fig = evoked.plot_joint()
        
        # --- 4. Create covariance --------------------------------------------
        """ TODO: make/call correct covariance function """
        
        # --- 5. Forward solution ---------------------------------------------
        #fwd = make_forward(subject_id, trans, evoked, subjects_dir=subjects_dir)
        
        
        # --- 6. Inverse solution ---------------------------------------------
        #stc, inv_op = make_inverse(subjects_dir, subject_id, fwd, evoked, noise_cov)
        
        
        # --- 7. Apply inverse to evokeds -------------------------------------
        #stc, inv_op = make_inverse(subjects_dir, subject_id, fwd, evoked, noise_cov)
        
        
        # --- 8. Save all files in BIDS structure -----------------------------
        #stc, inv_op = make_inverse(subjects_dir, subject_id, fwd, evoked, noise_cov)
        
        # --- 9. Generate and save MNE Report ---------------------------------
        
        
        
        
#### STEPS TO ADD
## ICA for eyeblinks in precprossessing 
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
        

