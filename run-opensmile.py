import ffmpeg # make sure to do pip install as pip install ffmpeg-python
import pandas as pd
import opensmile
from pathlib import Path
from tqdm import tqdm

"""
To do:
- upate the paths
- set the mode (single file for debugging, or folder for a regular run)

Want to check the exact OpenSMILE features? Use:

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals)
print(smile.feature_names)

To check the frame size and step:
grep -iE "frameSize|frameStep|frameMode" /home/userfolder/.local/lib/python3.13/site-packages/opensmile/core/config/gemaps/v01b/GeMAPSv01b_core.lld.conf.inc
grep -iE "reader.dmLevel|dmLevel=gemapsv01b_frame" /home/userfolder/.local/lib/python3.13/site-packages/opensmile/core/config/gemaps/v01b/GeMAPSv01b_core.lld.conf.inc
"""

# ================= File Setup =================

# 1. Update paths
single_file = "/"
folder = "/data/folder/path"
output_csv = "/filename.csv"

# 2. Set mode
SINGLE_FILE_MODE = False

# 3. Grab files
if SINGLE_FILE_MODE:
    files = [single_file]
else:
    files = [str(f) for f in Path(folder).glob("*.wav")] + \
            [str(f) for f in Path(folder).glob("*.mp3")]

# 4. Convert .mp3/m4a files -> wav
wav_files = []

for infile in tqdm(files, desc="Converting to WAV...", unit="file"): # adds a progress bar
    if infile.endswith('.mp3'):
        outfile = infile.replace('.mp3', '.wav')
    else:
        outfile = infile  # Already wav
    
    if not Path(outfile).exists():          # only convert if .wav doesn't already exist
        print(f"Converting: {infile}")
        (
            ffmpeg
            .input(infile)                   # Read .m4a file 
            .output(outfile, ac=1, ar=16000) # Write to .wav with sample rate: 44100 → 16000 Hz (matches the paper)
            .overwrite_output()              # Replace file if it exists
            .run(quiet=True)                 # Execute the conversion
        )
        print(f"Converted to {outfile}")
    else:
        print(f"Skipping {outfile} (already exists)")
    
    wav_files.append(outfile)

# ================= OpenSMILE Setup =================

# 1. Configure settings
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,      # Call the eGeMAPS feature set
    feature_level=opensmile.FeatureLevel.Functionals, # Give summary statistics e.g. mean, stddev, percentiles
)

# 2. Map the 88 eGeMAPS features to readable names for SHAP plot later

feature_mapping = {
    # ── Source Features ──────────────────────────────────────────────────────
    'Jitter Mean':                               'jitterLocal_sma3nz_amean',
    'Jitter Std Dev':                            'jitterLocal_sma3nz_stddevNorm',
    'Shimmer Mean':                              'shimmerLocaldB_sma3nz_amean',
    'Shimmer Std Dev':                           'shimmerLocaldB_sma3nz_stddevNorm',
    'HNR Mean':                                  'HNRdBACF_sma3nz_amean',
    'HNR Std Dev':                               'HNRdBACF_sma3nz_stddevNorm',
    'Harmonic Difference H1-H2 Mean':            'logRelF0-H1-H2_sma3nz_amean',
    'Harmonic Difference H1-H2 Std Dev':         'logRelF0-H1-H2_sma3nz_stddevNorm',
    'Harmonic Difference H1-A3 Mean':            'logRelF0-H1-A3_sma3nz_amean',
    'Harmonic Difference H1-A3 Std Dev':         'logRelF0-H1-A3_sma3nz_stddevNorm',
    'Loudness Mean':                             'loudness_sma3_amean',
    'Loudness Std Dev':                          'loudness_sma3_stddevNorm',
    'Loudness 20th Percentile':                  'loudness_sma3_percentile20.0',
    'Loudness 50th Percentile':                  'loudness_sma3_percentile50.0',
    'Loudness 80th Percentile':                  'loudness_sma3_percentile80.0',
    'Loudness Percentile Range (20th-80th)':     'loudness_sma3_pctlrange0-2',
    'Loudness Mean Rising Slope':                'loudness_sma3_meanRisingSlope',
    'Loudness Std Dev Rising Slope':             'loudness_sma3_stddevRisingSlope',
    'Loudness Mean Falling Slope':               'loudness_sma3_meanFallingSlope',
    'Loudness Std Dev Falling Slope':            'loudness_sma3_stddevFallingSlope',
    # Segment-Level
    'Loudness Peaks Per Second (Full Utterance)': 'loudnessPeaksPerSec',
    'Equivalent Sound Level (Full Utterance)':    'equivalentSoundLevel_dBp',

    # ── Filter Features ──────────────────────────────────────────────────────
    # Formant Frequency
    'F1 Frequency Mean':                         'F1frequency_sma3nz_amean',
    'F1 Frequency Std Dev':                      'F1frequency_sma3nz_stddevNorm',
    'F2 Frequency Mean':                         'F2frequency_sma3nz_amean',
    'F2 Frequency Std Dev':                      'F2frequency_sma3nz_stddevNorm',
    'F3 Frequency Mean':                         'F3frequency_sma3nz_amean',
    'F3 Frequency Std Dev':                      'F3frequency_sma3nz_stddevNorm',
    # Formant Bandwidth
    'F1 Bandwidth Mean':                         'F1bandwidth_sma3nz_amean',
    'F1 Bandwidth Std Dev':                      'F1bandwidth_sma3nz_stddevNorm',
    'F2 Bandwidth Mean':                         'F2bandwidth_sma3nz_amean',
    'F2 Bandwidth Std Dev':                      'F2bandwidth_sma3nz_stddevNorm',
    'F3 Bandwidth Mean':                         'F3bandwidth_sma3nz_amean',
    'F3 Bandwidth Std Dev':                      'F3bandwidth_sma3nz_stddevNorm',
    # Formant Amplitude
    'F1 Amplitude Mean':                         'F1amplitudeLogRelF0_sma3nz_amean',
    'F1 Amplitude Std Dev':                      'F1amplitudeLogRelF0_sma3nz_stddevNorm',
    'F2 Amplitude Mean':                         'F2amplitudeLogRelF0_sma3nz_amean',
    'F2 Amplitude Std Dev':                      'F2amplitudeLogRelF0_sma3nz_stddevNorm',
    'F3 Amplitude Mean':                         'F3amplitudeLogRelF0_sma3nz_amean',
    'F3 Amplitude Std Dev':                      'F3amplitudeLogRelF0_sma3nz_stddevNorm',
    # Spectral Flux
    'Spectral Flux (All Frames) Mean':           'spectralFlux_sma3_amean',
    'Spectral Flux (All Frames) Std Dev':        'spectralFlux_sma3_stddevNorm',
    'Spectral Flux (Voiced) Mean':               'spectralFluxV_sma3nz_amean',
    'Spectral Flux (Voiced) Std Dev':            'spectralFluxV_sma3nz_stddevNorm',
    'Spectral Flux (Unvoiced) Mean':             'spectralFluxUV_sma3nz_amean',
    # MFCC All Frames
    'MFCC1 (All Frames) Mean':                   'mfcc1_sma3_amean',
    'MFCC1 (All Frames) Std Dev':                'mfcc1_sma3_stddevNorm',
    'MFCC2 (All Frames) Mean':                   'mfcc2_sma3_amean',
    'MFCC2 (All Frames) Std Dev':                'mfcc2_sma3_stddevNorm',
    'MFCC3 (All Frames) Mean':                   'mfcc3_sma3_amean',
    'MFCC3 (All Frames) Std Dev':                'mfcc3_sma3_stddevNorm',
    'MFCC4 (All Frames) Mean':                   'mfcc4_sma3_amean',
    'MFCC4 (All Frames) Std Dev':                'mfcc4_sma3_stddevNorm',
    # MFCC Voiced
    'MFCC1 (Voiced) Mean':                       'mfcc1V_sma3nz_amean',
    'MFCC1 (Voiced) Std Dev':                    'mfcc1V_sma3nz_stddevNorm',
    'MFCC2 (Voiced) Mean':                       'mfcc2V_sma3nz_amean',
    'MFCC2 (Voiced) Std Dev':                    'mfcc2V_sma3nz_stddevNorm',
    'MFCC3 (Voiced) Mean':                       'mfcc3V_sma3nz_amean',
    'MFCC3 (Voiced) Std Dev':                    'mfcc3V_sma3nz_stddevNorm',
    'MFCC4 (Voiced) Mean':                       'mfcc4V_sma3nz_amean',
    'MFCC4 (Voiced) Std Dev':                    'mfcc4V_sma3nz_stddevNorm',
    # Alpha Ratio
    'Alpha Ratio (Voiced) Mean':                 'alphaRatioV_sma3nz_amean',
    'Alpha Ratio (Voiced) Std Dev':              'alphaRatioV_sma3nz_stddevNorm',
    'Alpha Ratio (Unvoiced) Mean':               'alphaRatioUV_sma3nz_amean',
    # Hammarberg Index
    'Hammarberg Index (Voiced) Mean':            'hammarbergIndexV_sma3nz_amean',
    'Hammarberg Index (Voiced) Std Dev':         'hammarbergIndexV_sma3nz_stddevNorm',
    'Hammarberg Index (Unvoiced) Mean':          'hammarbergIndexUV_sma3nz_amean',
    # Spectral Slope
    'Spectral Slope 0-500Hz (Voiced) Mean':      'slopeV0-500_sma3nz_amean',
    'Spectral Slope 0-500Hz (Voiced) Std Dev':   'slopeV0-500_sma3nz_stddevNorm',
    'Spectral Slope 0-500Hz (Unvoiced) Mean':    'slopeUV0-500_sma3nz_amean',
    'Spectral Slope 500-1500Hz (Voiced) Mean':   'slopeV500-1500_sma3nz_amean',
    'Spectral Slope 500-1500Hz (Voiced) Std Dev':'slopeV500-1500_sma3nz_stddevNorm',
    'Spectral Slope 500-1500Hz (Unvoiced) Mean': 'slopeUV500-1500_sma3nz_amean',

    # ── Prosodic-Volitional Features ─────────────────────────────────────────
    'F0 Mean':                                   'F0semitoneFrom27.5Hz_sma3nz_amean',
    'F0 Std Dev':                                'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
    'F0 20th Percentile':                        'F0semitoneFrom27.5Hz_sma3nz_percentile20.0',
    'F0 50th Percentile':                        'F0semitoneFrom27.5Hz_sma3nz_percentile50.0',
    'F0 80th Percentile':                        'F0semitoneFrom27.5Hz_sma3nz_percentile80.0',
    'F0 Percentile Range (20th-80th)':           'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2',
    'F0 Mean Rising Slope':                      'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope',
    'F0 Std Dev Rising Slope':                   'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope',
    'F0 Mean Falling Slope':                     'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope',
    'F0 Std Dev Falling Slope':                  'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope',
    # Segment-Level
    'Voiced Segments Per Second (Full Utterance)':                'VoicedSegmentsPerSec',
    'Mean Voiced Segment Length (Full Utterance)':                'MeanVoicedSegmentLengthSec',
    'Std Dev Voiced Segment Length (Full Utterance)':             'StddevVoicedSegmentLengthSec',
    'Mean Unvoiced Segment Length (Full Utterance)':              'MeanUnvoicedSegmentLength',
    'Std Dev Unvoiced Segment Length (Full Utterance)':           'StddevUnvoicedSegmentLength',
}

# ================= Feature Extraction =================

feature_set_all = []

for wav_file in tqdm(wav_files, desc="Extracting Features...", unit="file"):
    features = smile.process_file(wav_file)  # Extract ALL features
    # Rename columns to readable names
    features_renamed = features.rename(columns={v: k for k, v in feature_mapping.items()})
    features_renamed["file"] = Path(wav_file).name
    features_renamed["ParticipantID"] = int(Path(wav_file).name.split('_')[0]) # make sure we keep the participantIDs in, by creating this column, make int to keep order correct
    feature_set_all.append(features_renamed)

final_df = pd.concat(feature_set_all, ignore_index=True)

cols = ['ParticipantID'] + [c for c in final_df.columns if c not in ['ParticipantID', 'file']] # let's bring ParticipantID to 1st column, drop 'file' 
final_df = final_df[cols]

final_df.to_csv(output_csv, index=False)
print(f"\nDone! Saved {len(files)} file(s) with ALL 88 eGeMAPS features to csv")
