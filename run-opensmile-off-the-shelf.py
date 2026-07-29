import ffmpeg
import pandas as pd
import opensmile
from pathlib import Path
from tqdm import tqdm

"""
How to Run:
- install ffmpeg itself (not just the python package) e.g. 'brew install ffmpeg' for Mac
- pip install ffmpeg-python opensmile pandas tqdm
- upate paths
- set mode (single file for debugging, or folder)
- comment out or leave the Mapping sections in

Want to check the exact OpenSMILE features? Use:
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals)
print(smile.feature_names)
"""

# ================= File Setup =================
 
# 1. Update paths
single_file = "/path/to/one_file.mp3"      # only used if SINGLE_FILE_MODE = True
folder = "/path/to/your/audio_folder"      # a folder of .wav / .mp3 files
output_csv = "/path/to/save/features.csv"  # csv where results are saved
 
# 2. Set mode
SINGLE_FILE_MODE = False
 
# 3. Grabs files
if SINGLE_FILE_MODE:
    files = [single_file]
else:
    files = [str(f) for f in Path(folder).glob("*.wav")] + \
            [str(f) for f in Path(folder).glob("*.mp3")]
 
# 4. Convert .mp3 files -> .wav
wav_files = []
 
for infile in tqdm(files, desc="Converting to WAV...", unit="file"):
    if infile.endswith('.mp3'):
        outfile = infile.replace('.mp3', '.wav')
    else:
        outfile = infile  # already wav
 
    if not Path(outfile).exists():          # only convert if .wav doesn't already exist
        print(f"Converting: {infile}")
        (
            ffmpeg
            .input(infile)                   # read input file
            .output(outfile, ac=1, ar=16000) # write .wav, mono, 16000 Hz
            .overwrite_output()              # replace file if it exists
            .run(quiet=True)                 # run the conversion
        )
        print(f"Converted to {outfile}")
    else:
        print(f"Skipping {outfile} (already exists)")
 
    wav_files.append(outfile)
 
# ================= OpenSMILE Setup =================

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,      # eGeMAPS feature set
    feature_level=opensmile.FeatureLevel.Functionals, # summary stats: mean, stddev, percentiles
)
 
# ================= Feature Extraction =================
 
feature_set_all = []  # one result per file, stacked together at the end
 
for wav_file in tqdm(wav_files, desc="Extracting Features...", unit="file"):
    features = smile.process_file(wav_file)   # OpenSMILE returns one row of 88 features
    features["file"] = Path(wav_file).name    # keep track of which file this row came from
    feature_set_all.append(features)
 
# stack all rows into one table (one row per audio file)
final_df = pd.concat(feature_set_all, ignore_index=True)
 
final_df.to_csv(output_csv, index=False)
print(f"\nDone! Saved {len(final_df)} file(s) with all eGeMAPS features to csv")
