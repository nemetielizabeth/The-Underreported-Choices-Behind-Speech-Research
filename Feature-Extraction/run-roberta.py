from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm

# ---------------- File Setup ----------------
single_file = "/"
folder = "/path/to/trascripts/folder"
output_csv = "/filename.csv"

SINGLE_FILE_MODE = False

if SINGLE_FILE_MODE:
    files = [single_file]
else:
    files = [str(f) for f in Path(folder).glob("*.txt")]

# ---------------- Model Selection & Setup ----------------
# - "roberta-base"  -> 768-dim features
# - "roberta-large" -> 1024-dim features
# - "mental-roberta-base" -> 768-dim features (pretrained on mental health data)

model_name = "roberta-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

# add some GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check if GPU is available
model = model.to(device) # move the model's parameters onto the GPU memory
print(f"Using device: {device}")

# ---------------- Feature Extraction ----------------
# (!) Set layer
extraction_layer = 23

feature_set_all = []

for txt_file in tqdm(files, desc="Extracting Features...", unit="file"):
    text = Path(txt_file).read_text().strip() # read the transcript

    # ---------------- Sliding Window & Batching ----------------
    window_size = 512                        # tokens per window (RoBERTa's max)
    overlap = 256                            # token overlap between windows
    hop_size = window_size - overlap         # step between window starts

    # tokenize full text first (with no truncation)
    all_tokens = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = all_tokens["input_ids"][0]   # shape: [total_tokens]
    total_tokens = len(input_ids)            # total number of tokens in transcript

    # chop into overlapping windows
    windows = [
        input_ids[start : start + window_size]
        for start in range(0, total_tokens - window_size + 1, hop_size)]

    # if text is shorter than one window, just use the whole thing
    if len(windows) == 0:
        windows = [input_ids]

    all_embeddings = []
    for window in windows: # loop through each window
        inputs = {"input_ids": window.unsqueeze(0).to(device), # add batch dimension and move to GPU
                  "attention_mask": torch.ones_like(window).unsqueeze(0).to(device)}  # attend to all tokens

        with torch.no_grad(): # not training, no gradients needed
            outputs = model(**inputs, output_hidden_states=True)

        # Pool 1: average across tokens within each window -> one vector per window of [1024]
        window_vector = outputs.hidden_states[extraction_layer].mean(dim=1).squeeze()
        all_embeddings.append(window_vector) # store this window's vector

    # Pool 2: average across all windows -> one vector per transcript
    all_window_vectors = torch.stack(all_embeddings, dim=0) # [num_windows, 1024]
    pooled = all_window_vectors.mean(dim=0).cpu().numpy() # [1024]

    # ---------------- Build Output Row ----------------
    feature_cols = {f"roberta_{i}": pooled[i] for i in range(len(pooled))}  # 1024 features
    feature_cols["ParticipantID"] = int(Path(txt_file).name[4:8]) # extract 4-digit ID after CHK_ (for checked ones)
    #feature_cols["ParticipantID"] = int(Path(txt_file).name[:4]) #(for non-checked ones)
    feature_set_all.append(feature_cols) # add to results

# ---------------- Save Output ----------------
final_df = pd.DataFrame(feature_set_all)
cols = ['ParticipantID'] + [c for c in final_df.columns if c != 'ParticipantID']
final_df = final_df[cols]
final_df.to_csv(output_csv, index=False)
print(f"\nSaved shape {final_df.shape} as {output_csv}, which had {len(final_df)} participants and {len(pooled)} features.")
