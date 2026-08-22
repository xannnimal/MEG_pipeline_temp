# Copy this file to config.py and fill in your own paths.
# config.py is gitignored; each user will maintain their own local copy.
import os

###############################################################################
## --- User Specified Parameters -------------------------------------------------
# --- choose the subject modality, and task -----------------------------------
subject = 'S009'
task = 'VWFA'
modality = 'OPM' #'OPM' or 'CTF' or #'EEG'
# -- add your parent folder --
directory = '/Users/alexandria/Documents/STANFORD/DATA/2026_Gwilliams_MultimodalImaging/BIDS_test/'

# --- Set FreeSurfer path -----------------------------------------------------
subjects_dir = '/Users/alexandria/Downloads/freesurfer/subjects'
os.environ["SUBJECTS_DIR"] = subjects_dir

# --- Choose Visualization & Output -------------------------------------------
# if no extra bads to add to "info", leave empty
bads_runlog = []

## choose preprocessing
sss_bool = True ## Will do Foster's inverse with SSS, False=no preprocessing
msss_bool = False ## Will do foster's with mSSS, False=just with SSS

viz_bool = True
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
    trans = os.path.join(directory,f'{subject}/{mod}/sub-{subject}_task-{task}_{mod}_trans.{ext}')
    raw_files  = [f'sub-{subject}_task-{task}_{mod}_scan_raw.{ext}']
elif modality == 'CTF':
    mod= 'meg'
    ext='ds'
    trans = f'/Users/alexandria/Documents/STANFORD/DATA/2026_Gwilliams_MultimodalImaging/BIDS_test/{subject}/meg/sub-{subject}-ctf-trans.fif'
    if task=='VWFA' and (subject=='S001' or subject=='S009'):
        raw_files  = [[f'sub-{subject}_Task-{task}_{mod}_raw_01.{ext}',
                       f'sub-{subject}_Task-{task}_{mod}_raw_02.{ext}']]
    else:
        raw_files  = [f'sub-{subject}_Task-{task}_{mod}_raw.{ext}']
        
    
sample_dir = os.path.join(directory,f'{subject}/{mod}/')
dig_file   = None  # optional, set to path string if needed