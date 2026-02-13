# Pipeline development for C-SHARP MEG data

Load data (CTF and FieldLine FIF files) - make functions for each data type
Find events/triggers
Do coregistration, either in line or through YORC script for OPMs
Preprocessing - high-pass filter, add functions for SSS, homogenous field correction, etc
Make evokeds for each condition
Create covariance
Forward solution
Inverse solutions
Apply inverse to evokeds
Put into BIDS structure, specifically with events
Generate MNE-report
