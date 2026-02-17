# Pipeline development for C-SHARP MEG data

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
