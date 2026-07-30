import pandas as pd

"""
What this script does:
    Combines the best-performing audio and text feature csvs into two multimodal CSVs:
    1. Audio + Text (with labels)
    2. Audio + Text + Demographics (with labels)
    Both are ready for XGBoost classification (no need to put it through the prep script). 
"""

# ============== Input Files ==============
audio_df = pd.read_csv("audio/feature/set.csv")  
text_df = pd.read_csv("text/feature/set.csv")   
demo_df = pd.read_csv("demographic/feature/set.csv")

# ================= Shared Setup =================
label_cols = ['IDAS_MDD_Screening_Binary', 'IDAS_MDD_Balanced_Binary']
label_only = demo_df[['ParticipantID'] + label_cols]

audio_cols = [c for c in audio_df.columns if c != 'ParticipantID']
text_cols = [c for c in text_df.columns if c != 'ParticipantID']
demo_cols = [c for c in demo_df.columns if c not in ['ParticipantID'] + label_cols]

# ================= V1: Audio + Text =================
v1 = audio_df.merge(text_df, on="ParticipantID", how="inner")
v1 = v1.merge(label_only, on="ParticipantID", how="inner")
v1 = v1[['ParticipantID'] + label_cols + audio_cols + text_cols]

v1.to_csv("combined_audio_text.csv", index=False)
print(f"V1 (Audio + Text): {v1.shape}")

# ================= V2: Audio + Text + Demographics =================
v2 = audio_df.merge(text_df, on="ParticipantID", how="inner")
v2 = v2.merge(demo_df, on="ParticipantID", how="inner")
v2 = v2[['ParticipantID'] + label_cols + audio_cols + text_cols + demo_cols]

v2.to_csv("combined_audio_text_demog.csv", index=False)
print(f"V2 (Audio + Text + Demog): {v2.shape}")
