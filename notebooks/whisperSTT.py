!pip install -q openai-whisper

import os
import whisper
import csv
from tqdm import tqdm
from google.colab import drive

drive.mount('/content/drive')

model = whisper.load_model("medium")

folder_path = '/content/drive/My Drive/Lu_Speech'
mp3_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".mp3")])

output_csv = '/content/drive/My Drive/Lu_Speech_whisper_transcripts.csv'

if os.path.exists(output_csv):
    import pandas as pd
    done_df = pd.read_csv(output_csv)
    done_files = set(done_df["filename"])
else:
    done_files = set()

for fname in tqdm(mp3_files):
    if fname in done_files:
        continue
    file_path = os.path.join(folder_path, fname)

    try:
        result = model.transcribe(file_path, language="th")
        transcription = result["text"]
    except Exception as e:
        transcription = f"[ERROR] {e}"

    with open(output_csv, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text"])
        if os.stat(output_csv).st_size == 0:
            writer.writeheader()
        writer.writerow({"filename": fname, "text": transcription})
