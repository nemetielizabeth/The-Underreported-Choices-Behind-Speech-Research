"""
- This step follows immediately after Feature Extraction (either OpenSMILE, WavLM, RoBERTa, or LIWC-22). 
- Using the .csv with the extracted features as our input here, we:
    - (1) most importantly add our ground truth labels (Screening & Balanced cutoffs)
    - (2) decide whether to add demographic data or not
    - (3) for text files, decide whether to include all extracted features or only specific ones
    - (4) we clean up the file with a little rearranging to ready it for incoming classification
    - (5) sanity check we have all the features we actually want with a quick tally at the end
"""

# Setting Up ---------------------------------------------------------------------------------------

import pandas as pd
import os

INPUT_PATH = '/'
PARTICIPANT_INFO_PATH = '/'
OUTPUT_DIR = '/'

IS_LIWC_FILE         = False
INCLUDE_PUNCTUATION  = False  # LIWC only (x8)
INCLUDE_SUMMARY_VARS = False  # LIWC only (x8)
INCLUDE_DEMOGRAPHICS = False   # all files: add demographics e.g., age, sex, race, etc.

"""
Features that got extracted from LIWC, but get removed before going into model classification: 
- Punctuation features: All Punctuation, Periods, Commas, Question Marks, Exclamation Points, Apostrophes, Other Punctuation, Emojis
- Summary Variables: Total Word Count, Analytical thinking, Clout, Authentic, Emotional tone, Words per sentence, Big words, Dictionary words
"""

# Load Data ---------------------------------------------------------------------------------------

# grab our feature file, and the participant info file (contains labels + demographics)
input_stem = os.path.splitext(os.path.basename(INPUT_PATH))[0]
feature_df = pd.read_csv(INPUT_PATH)
start = feature_df.shape[1] - 1  # -1 for ParticipantID column
participant_info_df = pd.read_csv(PARTICIPANT_INFO_PATH)

# LIWC-specific Preparation ------------------------------------------------------------------------------

if IS_LIWC_FILE:
    feature_df['ParticipantID'] = feature_df['Filename'].str.extract(r'CHK_(\d{4})_').astype(int) # make ParticipantID column by pulling out subject ID from filename
    feature_df = feature_df.drop(columns=['Filename', 'Segment']) # now drop these
    start = feature_df.shape[1] - 1  # count our features, without including ID column

    if not INCLUDE_PUNCTUATION:
        feature_df = feature_df.drop(columns=['AllPunc', 'Period', 'Comma', 'QMark', 'Exclam', 'Apostro', 'OtherP', 'Emoji'], errors='ignore')

    if not INCLUDE_SUMMARY_VARS:
        feature_df = feature_df.drop(columns=['WC', 'Analytic', 'Clout', 'Authentic', 'Tone', 'WPS', 'BigWords', 'Dic', 'Linguistic'], errors='ignore')

# Organizing the Feature File  ------------------------------------------------------------------------------

# define column groups from participant_info for reordering
label_cols   = ['IDAS_MDD_Screening_Binary', 'IDAS_MDD_Balanced_Binary']  # always included
demo_cols    = ['AGE', 'SEX', 'SEXUAL_ORIENTATION_BINARY', 'RACE_BINARY', 'DISABILITY_BINARY',
                'Employment_Yes, 10 hours or less each week',
                'Employment_Yes, 11+ hours each week']  # optional, see INCLUDE_DEMOGRAPHICS

# merge to bring in labels (always) and demographics (optional)
merged = feature_df.merge(participant_info_df, on='ParticipantID', how='inner')

# everything that isn't an ID, label, or demographic = a feature
feature_cols = [c for c in merged.columns if c not in ['ParticipantID'] + label_cols + demo_cols]
feat_count = len(feature_cols) #(to keep track of how many features for output file naming)

# Demographics Preparation ----------------------------------------------------------------------------

if INCLUDE_DEMOGRAPHICS:
    merged   = merged[['ParticipantID'] + label_cols + demo_cols + feature_cols]
    demo_str = "7demog"
else:
    merged   = merged[['ParticipantID'] + label_cols + feature_cols]
    demo_str = "0demog"

# Save Set Up ---------------------------------------------------------------------------------------
"""
Expected shapes:
- eGeMAPS:  (73,  98)  = 1 ID + 2 labels + 88 features
- WavLM:    (73, 1027) = 1 ID + 2 labels + 1024 features
- roBERTa:  (73, 1027) = 1 ID + 2 labels + 1024 features
- LIWC:     (73, ~119) = 1 ID + 2 labels + 119 features
- +7 for adding demographic features
"""

if IS_LIWC_FILE:
    punc_str   = "with-punc" if INCLUDE_PUNCTUATION  else "no-punc"
    sumvar_str = "with-sumvar" if INCLUDE_SUMMARY_VARS else "no-sumvar"
    output_name = f'dataset_xgboost_{input_stem}_{punc_str}_{sumvar_str}_{feat_count}feats_{demo_str}.csv'
else:
    output_name = f'dataset_xgboost_{input_stem}_{feat_count}feats_{demo_str}.csv'

merged.to_csv(OUTPUT_DIR + output_name, index=False)

# tally for a nice visual sanity check
punc_delta = -8 if not INCLUDE_PUNCTUATION else 0
sumv_delta = -8 if not INCLUDE_SUMMARY_VARS else 0
demo_delta = +7 if INCLUDE_DEMOGRAPHICS else 0

print(f"Features:  {start}")
if IS_LIWC_FILE:
    print(f"                {punc_delta:+d}  (punctuation {'removed' if not INCLUDE_PUNCTUATION else 'included'})")
    print(f"                {sumv_delta:+d}  (summary vars {'removed' if not INCLUDE_SUMMARY_VARS else 'included'})")
print(f"                {demo_delta:+d}  (demographics {'added' if INCLUDE_DEMOGRAPHICS else 'not added'})")
print(f"                +2  (labels: Screening and Balanced Cutoff)")
print(f"                +1  (Participant ID)")
print(f"                ────")
final = start + (punc_delta + sumv_delta if IS_LIWC_FILE else 0) + demo_delta + 3
print(f"Final shape:    {final}  → {merged.shape}")
print(f"Saved: {output_name} to XGBOOST folder.")
