# Pipeline development for C-SHARP MEG data

C-SHARP MEG Forward & Inverse Pipeline
Supports both CTF (UCSF) and FieldLine OPM data.

## 1. Set up the YORC coregistration pipeline for FieldLine OPM Data
In lieu of using other coreg methods (HPI coils, Polhemus, etc.), coregister FieldLine OPM sensor locations with head, create `-trans.fif` file and add `trans` object to `raw.info` structure of data saved from FieldLine. See https://github.com/wadelab/yorc-gui/blob/master/USER_GUIDE.md for more details and options

### 1.A Install dependencies
- Python 3.11+
- `uv` for virtural environment management: https://docs.astral.sh/uv/ is recommended, but any manager will be fine
- FreeSurfer

### 1.B Clone and Enter the YORC coregistration Repository
```bash
git clone https://github.com/wadelab/yorc-gui.git
cd yorc-gui
```
can also clone using GitHub Desktop with this link: https://github.com/wadelab/yorc-gui.git

### 1.C Create Environment and Install Dependencies
Use `uv` to manage virtual environments. While still in `yorc-gui` cloned folder, create `venv`:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

Notes:
- `requirements.txt` pins compatible versions (including SciPy for MNE compatibility).
- `uv pip install -e .` installs the local `yorc` package in editable mode.

### 1.D FreeSurfer scalp surface to STL conversion
Generate `mriscalp.stl` and relevent files for forward and inverse modeling

Always run below (as example) to make sure Freesurfer is setup correctly
```bash
export FREESURFER_HOME=/Applications/freesurfer/8.1.0
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```

Then, if you already ran `recon-all`:

```bash
export SUBJECTS_DIR=/path/to/subjects_dir
subject=subj01

# Create scalp/head surface if needed
mkheadsurf -subjid ${subject}

# Convert surface to STL
mris_convert ${SUBJECTS_DIR}/${subject}/surf/lh.seghead \
             ${SUBJECTS_DIR}/${subject}/surf/mriscalp.stl
```

### 1.E Prepare Input Files

You should now have these 4 files:
- `inside` LIDAR mesh/point cloud (`.ply/.stl/.obj/.pcd`)  
  Example: scan with participant in helmet
- `outside` LIDAR mesh/point cloud (`.ply/.stl/.obj/.pcd`)  
  Example: scan without helmet
- `mri scalp` surface (`.stl` or `.fif` or mesh formats accepted by GUI)
- `meg` raw FIF (`.fif`)

## 3. Do YORC coregistration with Lidar camera scans
Generate -trans.fif and add trans object to raw data info structure by running coregistration code. NOTE: this will overwite the original raw file to include the `trans` object, reccommended to save a copy of original raw, umodified, in a separate place.

To run `YORC.py` or `manual_YORC.py`, you will be prompted to pick points on each of the 4 files used to coreg them all together. First, `cd ..` out to your home or user directory, then run:

```bash
python3 /"your_path_here"/manual_YORC.py
  -om /"your_path_here"/subID_01_R.ply
  -im /"your_path_here"/subID_02_H.ply
  -s /"your_path_here"/mriscalp.stl
  -m /"your_path_here"/subID-raw.fif -lm 1 2 3 4 5 6
```

## 4. Run pipeline
1. Load data (CTF vs FIF files). For FieldLine OPM data, be sure to load the updated raw data with `trans` object added
2. Preprocessing - Reject eyeblinks with ICA, high-pass filter, add functions for SSS, homogenous field correction, etc
3. Make evokeds for each condition
4. Create covariance
5. Forward solution
6. Inverse solutions - explore options. Most take norm and throw away the orientation. Look into “free” orientation vs locked
7. Apply inverse to evokeds
8. Put into BIDS structure, specifically with events and evoked.fif files
9. Generate MNE-report
