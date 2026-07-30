import ffmpeg
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

"""
WavLM Large outputs 1024 features per frame (vs. OpenSMILE's 88 handcrafted features).
Unlike OpenSMILE, these features have no human-interpretable names --
they are learned embeddings from pretraining on 94k hrs of speech data.
We apply a sliding window (2s window, 1s hop), then average-pool across windows, 
to produce one 1024-dim vector per recording.
"""

#---------------- File Setup ----------------

# 1. Update paths
single_file = "/"
folder = "/folder/to/aata"
output_csv = "/filename.csv"

# 2. Set mode
SINGLE_FILE_MODE = False

# 3. Grab files
if SINGLE_FILE_MODE:
    files = [single_file]
else:
    # change pattern if you want "*.wav" or "*.mp3" or "*.m4a"
    files = [str(f) for f in Path(folder).glob("*.wav")] + \
            [str(f) for f in Path(folder).glob("*.mp3")]

# 4. Convert .mp3/m4a files -> wav (same as OpenSMILE pipeline)
wav_files = []

for infile in tqdm(files, desc="Converting to WAV...", unit="file"):
    if infile.endswith('.mp3'):
        outfile = infile.replace('.mp3', '.wav')
    else:
        outfile = infile  # Already wav
    
    if not Path(outfile).exists():
        print(f"Converting: {infile}")
        (
            ffmpeg
            .input(infile)
            .output(outfile, ac=1, ar=16000) # 16 kHz to match WavLM expectations
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"Converted to {outfile}")
    else:
        # if input is mp3 and wav already exists, we still want to use the existing wav
        print(f"Skipping {outfile} (already exists)")
    
    wav_files.append(outfile)

# ---------------- Model Selection ----------------

# 1. Choose the model
# - "microsoft/wavlm-base"  -> 768-dim features
# - "microsoft/wavlm-base-plus" -> 768-dim features (pretrained on more data)
# - "microsoft/wavlm-large" -> 1024-dim features

model_name = "microsoft/wavlm-large"

# 2. Load processor and model
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name) # preprocessor
model = WavLMModel.from_pretrained(model_name) # download pre-trained model
model.eval() # inference mode
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters") # let's check it downloaded

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check if GPU is available
model = model.to(device) # move the model's 300M+ parameters onto the GPU memory
print(f"Using device: {device}")

# ---------------- Feature Extraction ----------------

# (!) Set layer
extraction_layer = 23

feature_set_all = []

for wav_file in tqdm(wav_files, desc="Extracting Features...", unit="file"):
    waveform, sample_rate = torchaudio.load(wav_file) # Load audio
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000) # Resample to 16kHz if any file slipped through
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0).numpy() # processor expects 1D numpy array, not 2D tensor

# ---------------- Sliding Window & Batching ----------------
    # Sliding window: 2-second windows, 1-second overlap (matches OpenSMILE windowing)
    window_size = 2 * 16000              # 32000 samples (2-second window)
    overlap     = 1 * 16000              # 16000 samples (1-second overlap)
    hop_size    = window_size - overlap  # 16000 samples (step between window starts)

    # each 2-second window is one sample; partial windows at boundaries are dropped
    windows = [
        waveform[start : start + window_size]
        for start in range(0, len(waveform) - window_size + 1, hop_size)
    ]

    # Process windows in chunks of 128 to avoid GPU memory overflow
    batch_size = 128
    all_embeddings = []  # will collect one [1024] vector per window

    for i in range(0, len(windows), batch_size):
        batch = windows[i : i + batch_size]
        inputs = processor(batch, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():  # not training, so don't need to track gradients
            outputs = model(**inputs, output_hidden_states=True)

        # Shape: [batch_size, time_steps, 1024]
        batch_embeddings = outputs.hidden_states[extraction_layer]

        # Pool 1: average across time frames within each window
        # [batch_size, time_steps, 1024] -> [batch_size, 1024]
        window_vectors = batch_embeddings.mean(dim=1)
        all_embeddings.append(window_vectors)

    # Pool 2: average across all windows -> one vector per recording
    # [total_windows, 1024] -> [1024]
    all_window_vectors = torch.cat(all_embeddings, dim=0)  # stack all batches
    pooled = all_window_vectors.mean(dim=0).cpu().numpy()  # final [1024] vector

    # ---------------- Build Output Row ----------------
    # Mirror OpenSMILE output structure: ParticipantID + 1024 features
    feature_cols = {f"wavlm_{i}": pooled[i] for i in range(len(pooled))}
    feature_cols["ParticipantID"] = int(Path(wav_file).name.split('_')[0])
    feature_set_all.append(feature_cols)

final_df = pd.DataFrame(feature_set_all)

# ---------------- Prep Output File ----------------
cols = ['ParticipantID'] + [c for c in final_df.columns if c != 'ParticipantID'] # Bring ParticipantID to 1st column
final_df = final_df[cols]
final_df.to_csv(output_csv, index=False)
print(f"\nDone! Saved {len(files)} file(s) with {len(pooled)} WavLM features to csv")
