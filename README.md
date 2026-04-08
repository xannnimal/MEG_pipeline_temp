# Pipeline development for C-SHARP MEG data

C-SHARP MEG Forward & Inverse Pipeline
Supports both CTF (UCSF) and FieldLine OPM data.

## 1. FreeSurfer Setup & MRI Data Preprocessing
Generate FreeSurfer-formatted surfaces from T1-weighted MRI scan `T1_<subject_id>.nii.gz` that are suitable for usage of MNE-Python

[**FreeSurfer**](https://surfer.nmr.mgh.harvard.edu/) is a tool that:
- Segments brain tissue (e.g., pial, white matter, cortex)
- Reconstructs cortical surfaces
- Outputs surface files with vertex coordinates in `(x, y, z)` space

### 1.A Setting Up FreeSurfer Environment (macOS/Linux) - SKIP if already set up!

**STEP 1: To make FreeSurfer accessible in your terminal:**
1. Open your shell configuration file (e.g., `.zshrc` or `.bashrc`):
   ```bash
   nano ~/.zshrc  # or use ~/.bashrc if you're using bash
2. Add the following lines at the end of the file (change the path to your local FreeSurfer directory):
    ```bash
    export FREESURFER_HOME=/Users/qyfeng/Downloads/freesurfer
    source $FREESURFER_HOME/SetUpFreeSurfer.sh
3. Reload your shell configuration:
    ```bash
    source ~/.zshrc  # or ~/.bashrc
4. confirm that FreeSurfer is correctly installed:
    ```bash
    recon-all -version

**STEP 2: Register for a FreeSurfer License**
1. Go to the official FreeSurfer registration page: [http://surfer.nmr.mgh.harvard.edu/registration.html](http://surfer.nmr.mgh.harvard.edu/registration.html)

2. Enter your information and download the `license.txt` file.

3. Save the file to your `$FREESURFER_HOME` directory (the same path you exported earlier).  
   For example:
   ```bash
   mv ~/Downloads/license.txt $FREESURFER_HOME

Now, FreeSurfer environment has been successfully setup! 

### 1.B Use FreeSurfer commands to process anatomical data
takes the raw T1 NIfTI, and runs full cortical reconstruction pipeline:
  ```bash
  recon-all -s [subject_id] -i /path/to/T1.nii.gz -all
  ```
Once complete, FreeSurfer will output surface and volume files under:
  ```bash
  $SUBJECTS_DIR/<subject_id>/
  ```
> **Note:** The full FreeSurfer processing (`recon-all`) can take 6–8 hours to complete, depending on your machine.

### 1.C FreeSurfer scalp surface to STL conversion
Generate `mriscalp.stl` used in coregsitration with the 3D RevoPoint scanner results and MRI

Always run below (as example) to make sure Freesurfer is setup correctly
```bash
export FREESURFER_HOME=/Applications/freesurfer/8.1.0
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```

Then, if you already ran `recon-all`:

```bash
export SUBJECTS_DIR=/path/to/subjects_dir
subject=subj01

# Create scalp/head surface if needed, this will generate lh.seghead and rh.seghead
mkheadsurf -subjid ${subject}

# Convert surface to STL
mris_convert ${SUBJECTS_DIR}/${subject}/surf/lh.seghead \
             ${SUBJECTS_DIR}/${subject}/surf/mriscalp.stl
```
### 1.D Run MNE Watershed to create BEM files
These files are used for forward and inverse modeling
```bash
export SUBJECTS_DIR=/path/to/subjects_dir
subject=subj01

mne watershed_bem -d $SUBJECTS_DIR -s subject --overwrite
```
Files will save in ```SUBJECTS_DIR/subject/bem```

### (Optional) 1.E Processing FLASH images

from a FLASH scan (in our case, it is a FSPGR - Fast Spoiled Gradient Echo, Multi-Echo), we acquire two files: 1 ``.dicom.zip``, 1 ``.nii.gz``. They are used for bias field correction and improve surface reconstruction, which is usually better than using T1 alone.

**STEP1: Inspect the number of echoes, using ``FSL`` package to check raw NifTi files:**

0. download FSL on a Linux machine:
```bash
curl -Ls https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/getfsl.sh | sh -s
```
1. check FLASH file dimension
```bash
fslhd FSPGR_ME_8Echoes/*.nii.gz
```
look for the ``dim4``. If it is 4D, we will need to split it via

**STEP2: Split into single echoes, generating multiple ``NifTi`` files:**
```bash
fslsplit FSPGR_ME_8Echoes/yourfile.nii.gz flash_
```
output are flash_000*.nii.gz in your pwd, each one corresponds to a single echo (or flip angle)


## 2. Set up the YORC coregistration pipeline for FieldLine OPM Data
In lieu of using other coreg methods (HPI coils, Polhemus, etc.), coregister FieldLine OPM sensor locations with head, create `-trans.fif` file and add `trans` object to `raw.info` structure of data saved from FieldLine. See [Wadelab/yorc-gui/User_Guide](https://github.com/wadelab/yorc-gui/blob/master/USER_GUIDE.md) for more details and options

### 2.A Install dependencies
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for virtural environment management is recommended, but any manager will be fine
- FreeSurfer

### 2.B Clone and Enter the YORC coregistration Repository
```bash
git clone https://github.com/wadelab/yorc-gui.git
cd yorc-gui
```
can also clone using GitHub Desktop with this link: [https://github.com/wadelab/yorc-gui.git](https://github.com/wadelab/yorc-gui.git)

### 2.C Create Environment and Install Dependencies
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

### 2.D Prepare Input Files

You should now have these 4 files:
- `inside` LIDAR mesh/point cloud (`.ply/.stl/.obj/.pcd`)  
  Example: scan with participant in helmet
- `outside` LIDAR mesh/point cloud (`.ply/.stl/.obj/.pcd`)  
  Example: scan without helmet
- `mri scalp` surface (`.stl` reccomended or `.fif` or mesh formats accepted by GUI)
- `meg` raw FIF (`.fif`)

## 3. Do YORC coregistration with Lidar camera scans
Generate -trans.fif and add trans object to raw data info structure by running coregistration code. 

> **Note:** This process will overwite the original raw file to include the `trans` object. It is reccommended to save a copy of original, unmodified raw file(s)

In the terminal, navigate to the cloned `yorc-gui` folder we created earlier and activate the virtual environment:
```bash
source .venv/bin/activate
```

Open York GUI for more visuals and troubleshooting. In terminal, `cd ..` back to "User/<name>/", then launch GUI:
```bash
uv run yorc-tripanel-gui
```
Then, select the same four files (with path names described from "User/<name>/", eg. "documens/folder") as above and follow the steps as prompted. These will ask you to pick the 7 target landmarks on the sensor helmet and various fiducials to math the inside/outside scan with the subeject MRI. Preview sensors in "fast mode" first before calculating more stable solution. Finally, "apply to fif" will save the `trans` to files

More information and instruction at [**GUI usage**](https://github.com/wadelab/yorc-gui/blob/master/GUI_USAGE.md)

### (Alternative) 3.A
Transformation matricies can also be found using `manual_YORC.py` in terminal instead of GUI. Do this method if the scan is missing 1 of the 7 landmarks on the helmet! You will be prompted to pick points on each of the 4 files used to coreg them all together. After activiating the environment, `cd ..` out to your home or user directory, then run:

```bash
python3 /"your_path_here"/manual_YORC.py
  -om /"your_path_here"/subID_01_R.ply
  -im /"your_path_here"/subID_02_H.ply
  -s /"your_path_here"/mriscalp.stl
  -m /"your_path_here"/subID-raw.fif
  -lm 1 2 3 4 5 6 7
```
> **Note:** `-lm 1 2 3 4 5 6 7` specifices the landmarks visibile on the helmet Lidar scan. For example, if the first landmark on the right of the scan isn't visible, run the code with `-lm 2 3 4 5 6 7`

Follow prompts in terminal
- `Shift + Left Click`: add point
- `Shift + Right Click`: remove last point
- `q`: move to next step
- check overlays of in/out scan with MRI scalp surface and sensor locations in visual popups before applying to `.fif`

More information and installation instructions can be found at [York OPM Registration Code](https://vcs.ynic.york.ac.uk/ynic-public/yorc)

> **Note:** Manual YORC python requires Python3.11 and Scipy<1.17 to run. 

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
