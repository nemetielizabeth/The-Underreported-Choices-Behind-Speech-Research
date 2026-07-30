from scipy.io import wavfile
import subprocess
import os
from pathlib import Path

# ============================================================
CROP_DURATION_SECONDS = 300 # set max duration
START_SECOND = 0
BORDERLINE_THRESHOLD = 290  # set passable threshold (keep it only to a few seconds). e.g. for 300 second cutoff, 297 is still acceptable
INPUT_DIR = "audio/data/folder/path"
OUTPUT_DIR = "output/folder/path"
MODE = "cut" # set 
# ============================================================

def format_duration(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"

# Step 1: Convert any non-wav files to wav
print("Step 1: Converting to wav...")
all_files = os.listdir(INPUT_DIR)
for filename in all_files:
    if filename.endswith(('.mp3', '.m4a')) and not filename.startswith('.'):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(INPUT_DIR, filename.rsplit('.', 1)[0] + '.wav')
        if not os.path.exists(output_path):
            subprocess.run([
                'ffmpeg', '-y', '-i', input_path,
                '-acodec', 'pcm_s16le', '-ar', '16000', output_path
            ], capture_output=True)
            print(f"  Converted: {filename}")
print("Conversion done.\n")

# Step 2: Crop wav files
print("Step 2: Cropping...")
files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')]
required_sec = START_SECOND + CROP_DURATION_SECONDS
borderline_min = START_SECOND + BORDERLINE_THRESHOLD

too_short = []
borderline = []

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

for filename in files:
    sample_rate, audio = wavfile.read(os.path.join(INPUT_DIR, filename))
    duration_sec = len(audio) / sample_rate
    
    if duration_sec < borderline_min:
        if MODE == "notify":
            raise ValueError(f"{filename} too short: {format_duration(duration_sec)}")
        too_short.append((filename, duration_sec))
        continue
    elif duration_sec < required_sec:
        borderline.append((filename, duration_sec))
        wavfile.write(os.path.join(OUTPUT_DIR, filename), sample_rate, audio)
    else:
        start_sample = int(START_SECOND * sample_rate)
        end_sample = int((START_SECOND + CROP_DURATION_SECONDS) * sample_rate)
        wavfile.write(os.path.join(OUTPUT_DIR, filename), sample_rate, audio[start_sample:end_sample])

# Summary
if too_short:
    print(f"\nExcluded {len(too_short)} files (under {BORDERLINE_THRESHOLD}s):")
    for f, d in too_short:
        print(f"  - {f}: {format_duration(d)}")

if borderline:
    print(f"\nBorderline {len(borderline)} files (included as-is):")
    for f, d in borderline:
        print(f"  - {f}: {format_duration(d)}")

print(f"\nDone. Cropped {len(files) - len(too_short)} files to {OUTPUT_DIR}")
