import pandas as pd
from jiwer import wer, cer

# Read the ASR results
df = pd.read_csv("results/asr_transcripts.csv")

# Compare reference transcripts with Whisper outputs
for _, row in df.iterrows():
    if pd.isna(row["reference"]) or pd.isna(row["whisper"]):
        continue

    reference = str(row["reference"])
    hypothesis = str(row["whisper"])

    print(f"\nClip: {row['clip_id']}")
    print(f"Language: {row['language']}")
    print(f"WER: {wer(reference, hypothesis):.4f}")
    print(f"CER: {cer(reference, hypothesis):.4f}")