from df.enhance import enhance, init_df, load_audio
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import ffmpeg

"""
We'll be using Deepfilternet to denoise audios by Schroter, Hendrik, et al. 2022.

How to run:
    sbatch denoise.sh - to run this file
"""

# --------------- File Setup ---------------

# 1. Update paths
single_file = "/"
folder = "/"
output_folder = "/"
Path(output_folder).mkdir(exist_ok=True)

# 2. Set mode & grab files
SINGLE_FILE_MODE = False

if SINGLE_FILE_MODE:
    files = [single_file]
else:
    files = list(Path(folder).glob("*.wav")) + list(Path(folder).glob("*.mp3"))

# --------------- Mp3 -> Wav Conversion if needed ---------------

wav_files = []
for infile in tqdm(files, desc="Converting to WAV...", unit="file"):
    infile = str(infile)
    if infile.endswith('.mp3'):
        outfile = str(Path(output_folder) / Path(infile).stem) + '.wav'
        if not Path(outfile).exists():
            (
                ffmpeg
                .input(infile)
                .output(outfile, ac=1, ar=48000)  # 48kHz to match DeepFilterNet
                .overwrite_output()
                .run(quiet=True)
            )
        wav_files.append(outfile)
    else:
        wav_files.append(str(infile))

# --------------- Denoising ---------------

# Load model
print("Loading DeepFilterNet model...")
model, df_state, _ = init_df()

for audio_file in tqdm(wav_files, desc="Denoising", unit="file"):
    # Load audio
    audio, sr = load_audio(audio_file, sr=df_state.sr())

    # Denoising part with enhance function
    enhanced = enhance(model, df_state, audio)

    # Convert and save
    enhanced_np = enhanced.cpu().numpy().T
    filename = Path(audio_file).stem + "_denoised.wav"
    output_path = Path(output_folder) / filename
    sf.write(str(output_path), enhanced_np, df_state.sr())
    
    # delete any temp wav files from final folder
    if audio_file.startswith(str(output_folder)) and not audio_file.endswith('_denoised.wav'):
        Path(audio_file).unlink()

print(f"\nDone! Denoised files saved to: {output_folder}")
