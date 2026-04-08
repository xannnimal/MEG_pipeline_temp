# Copy this file to config.py and fill in your own paths.
# config.py is gitignored; each user will maintain their own local copy.

# --- Data paths --------------------------------------------------------------
sample_dir = '/path/to/your/meg/data/'
raw_files  = ['your_raw_file.fif']
trans      = '/path/to/your/trans.fif'
dig_file   = None  # optional, set to path string if needed

# --- FreeSurfer --------------------------------------------------------------
subjects_dir = '/path/to/freesurfer/subjects/'
subject      = 'S001'

# --- Visualization & Output --------------------------------------------------------------
viz_bool = True
save_report = True
