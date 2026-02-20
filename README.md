# Pipeline development for C-SHARP MEG data

C-SHARP MEG Forward & Inverse Pipeline
Supports both CTF (UCSF) and FieldLine OPM data.

## FreeSurfer
Generate mriscalp.stl and relevent files for forward and inverse modeling
TODO: add specifics

## Coregistration with Lidar camera
Generate -trans.fif and add trans object to raw data info structure by running coregistration code. TODO: add specifics

## Run pipeline.
1. Load data (CTF vs FIF files)
2. Preprocessing - Reject eyeblinks with ICA (?), high-pass filter, add functions for SSS, homogenous field correction, etc
3. Make evokeds for each condition
4. Create covariance
5. Forward solution
6. Inverse solutions - explore options. Most take norm and throw away the orientation. Look into “free” orientation vs locked
7. Apply inverse to evokeds
8. Put into BIDS structure, specifically with events and evoked.fif files
9. Generate MNE-report
