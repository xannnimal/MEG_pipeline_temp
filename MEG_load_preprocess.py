#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 10:40:01 2026

@author: alexandria


Pipeline development for C-SHARP MEG data, both CTF data from USCF and FieldLine OPM

Load data (CTF vs FIF files) - make functions for each data type
Do coregistration, either in line or through YORC script
Preprocessing - Reject eyeblinks with ICA (?), high-pass filter, add functions for SSS, homogenous field correction, etc
Make evokes for each condition
Create covariance
Forward solution
Inverse solutions - explore options. Most take norm and throw away the orientation. Look into “free” orientation vs locked
Apply inverse to evokeds
Put into BIDS structure, specifically with events
Generate MNE-report
"""

