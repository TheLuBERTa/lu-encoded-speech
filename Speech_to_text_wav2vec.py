!pip install transformers datasets torchaudio
!pip install pydub

import os
import torch
import pandas as pd
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio
from tqdm import tqdm

# STT ภาษาไทย
model_name = "airesearch/wav2vec2-large-xlsr-53-th"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model.eval()

folder_path = '/content/drive/My Drive/Lu_Speech'
mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]

results = []

# STT
for fname in tqdm(mp3_files):
    try:
        # Step 1: load mp3 → wav (in memory)
        mp3_path = os.path.join(folder_path, fname)
        audio = AudioSegment.from_mp3(mp3_path).set_frame_rate(16000).set_channels(1)
        temp_wav = "/content/temp.wav"
        audio.export(temp_wav, format="wav")

        # Step 2: load waveform
        waveform, sample_rate = torchaudio.load(temp_wav)

        # Step 3: prepare input
        inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)

        # Step 4: inference
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(pred_ids)[0]

        # save
        results.append({"filename": fname, "text": transcription})

    except Exception as e:
        results.append({"filename": fname, "text": f"[ERROR] {e}"})
        continue

df = pd.DataFrame(results)

df.to_csv("/content/drive/My Drive/Lu_Speech_transcriptions.csv", index=False)
