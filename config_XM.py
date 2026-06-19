# Copy this file to config.py and fill in your own paths.
# config.py is gitignored; each user will maintain their own local copy.
import os

###############################################################################
## --- User Specified Parameters -------------------------------------------------
# --- choose the subject modality, and task -----------------------------------
subject = 'S009'
task = 'VWFA'
modality = 'CTF' #'OPM' or 'CTF' or #'EEG'
# -- add your parent folder --
directory = '/Users/alexandria/Documents/STANFORD/DATA/2026_Gwilliams_MultimodalImaging/BIDS_test/'

# --- Set FreeSurfer path -----------------------------------------------------
subjects_dir = '/Users/alexandria/Downloads/freesurfer/subjects'
os.environ["SUBJECTS_DIR"] = subjects_dir

# --- Choose Visualization & Output -------------------------------------------
viz_bool = True
sss_bool = False
save_report = True
save_raw = False
if save_report:
    # specify output directory to save reports
    report_dir='/Users/alexandria/Documents/STANFORD/DATA/2026_Gwilliams_MultimodalImaging/reports/'


###############################################################################
# --- autofill Data paths -----------------------------------------------------
if modality == 'OPM':
    mod = 'meg'
    ext='fif'
    trans = os.path.join(directory,f'{subject}/{mod}/sub-{subject}_task-{task}_{mod}2_trans.{ext}')
elif modality == 'CTF':
    mod= 'meg'
    ext='ds'
    trans = f'/Users/alexandria/Documents/STANFORD/DATA/2026_Gwilliams_MultimodalImaging/BIDS_test/{subject}/meg/sub-{subject}-ctf-trans.fif'

sample_dir = os.path.join(directory,f'{subject}/{mod}/')

if task=='VWFA' and modality == 'CTF':
    raw_files  = [[f'sub-{subject}_task-{task}_{mod}_raw_01.{ext}',
                   f'sub-{subject}_task-{task}_{mod}_raw_02.{ext}']]
else:
    raw_files  = [f'sub-{subject}_task-{task}_{mod}_raw.{ext}']
dig_file   = None  # optional, set to path string if needed


