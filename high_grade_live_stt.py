# high_grade_transcriber.py
# ========================================================
# HIGH-GRADE TRANSCRIPTION ENGINE (Works Offline + Colab)
# Run with: python high_grade_transcriber.py
# Requirements already in your requirements.txt:
# faster-whisper, sounddevice, numpy, torch, torchaudio, librosa, tqdm

import os
import queue
import threading
import sys
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
import librosa
from tqdm import tqdm

# Auto-detect Colab (safe on local machine)
try:
    from google.colab import files

    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    files = None


class HighGradeTranscriber:
    def __init__(self, model_size="auto", device="auto", compute_type=None):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_size == "auto":
            model_size = "medium" if device == "cuda" else "base"
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        self.device = device
        self.model_size = model_size
        self.compute_type = compute_type
        self.sample_rate = 16000

        print(f"\n🔥 HIGH-GRADE TRANSCRIPTION ENGINE LOADED")
        print(f"   • Device       : {device.upper()}")
        print(f"   • Model        : {model_size} ({compute_type})")
        print(f"   • Optimizations: RMS + Dynamic filter + VAD tuned\n")

        print("Loading model...", end=" ")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("done!\n")

        self.initial_prompt = (
            "conversation, accurate, English, clear speech, natural dialogue"
        )
        self.transcript_file = "high_grade_transcript.txt"
        with open(self.transcript_file, "w", encoding="utf-8") as f:
            f.write("")

        # Live mode variables
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.audio_buffer = []

    # ====================== CORE TRANSCRIPTION (same as before) ======================
    def _rms_normalize(self, audio):
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio / rms * 0.5
        return audio.astype(np.float32)

    def _chunk_long_audio(self, audio, sr=16000, chunk_sec=25, overlap_sec=5):
        chunk_samples = int(chunk_sec * sr)
        overlap_samples = int(overlap_sec * sr)
        step = chunk_samples - overlap_samples
        chunks = []
        for start in range(0, len(audio), step):
            chunk = audio[start : start + chunk_samples]
            if len(chunk) >= sr * 3:
                chunks.append(chunk)
        return chunks

    def _transcribe_chunk(self, audio_np):
        audio_np = self._rms_normalize(audio_np)
        segments, _ = self.model.transcribe(
            audio_np,
            beam_size=5,
            temperature=0.0,
            language="en",
            vad_filter=True,
            vad_parameters=dict(threshold=0.4, min_silence_duration_ms=500),
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
        )
        text_parts = []
        for seg in segments:
            thresh = -1.6 if len(seg.text) > 30 else -2.0
            if seg.avg_logprob > thresh and seg.no_speech_prob < 0.6:
                text_parts.append(seg.text.strip())
        return " ".join(text_parts).strip()

    def transcribe_file(self, audio_path):
        print(f"\n📂 Processing: {os.path.basename(audio_path)}")
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        audio = self._rms_normalize(audio)
        duration = len(audio) / sr
        print(f"   Duration: {duration:.1f} s")

        if duration > 30:
            print("   → Chunking long audio...")
            chunks = self._chunk_long_audio(audio, sr)
            text_parts = []
            pbar = tqdm(chunks, desc="   ", unit="chunk")
            for chunk in pbar:
                part = self._transcribe_chunk(chunk)
                if part:
                    text_parts.append(part)
            full_text = " ".join(text_parts)
        else:
            full_text = self._transcribe_chunk(audio)

        full_text = full_text.replace("live captioning conversation", "").strip()

        print("\n" + "─" * 90)
        print(f"TRANSCRIPTION → {os.path.basename(audio_path)}")
        print("─" * 90)
        print(full_text)
        print("─" * 90 + "\n")

        with open(self.transcript_file, "a", encoding="utf-8") as f:
            f.write(f"\n===== {os.path.basename(audio_path)} =====\n{full_text}\n")

        return full_text

    # ====================== LIVE MICROPHONE STREAMING (Offline only) ======================
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        chunk = indata[:, 0].copy()
        self.audio_buffer.extend(chunk)
        needed = int(3.0 * self.sample_rate)  # 3-second chunks
        if len(self.audio_buffer) >= needed:
            to_process = np.array(self.audio_buffer[:needed], dtype=np.float32)
            self.audio_queue.put(to_process)
            self.audio_buffer = self.audio_buffer[needed:]

    def process_live(self):
        while self.is_running:
            try:
                audio = self.audio_queue.get(timeout=1)
                text = self._transcribe_chunk(audio)
                if text and len(text) > 2:
                    if "live captioning" in text.lower() and len(text.split()) < 8:
                        continue
                    print(f"🗣  {text}")
                    sys.stdout.flush()
                    with open(self.transcript_file, "a", encoding="utf-8") as f:
                        f.write(text + " ")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Live error: {e}")

    def start_live(self):
        print("\n🎤 LIVE MICROPHONE MODE STARTED")
        print("Speak now... (Ctrl+C to stop)\n")
        self.is_running = True

        thread = threading.Thread(target=self.process_live, daemon=True)
        thread.start()

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.2),
            ):
                while self.is_running:
                    sd.sleep(1000)
        except KeyboardInterrupt:
            self.is_running = False
            print("\n\n🛑 Live transcription stopped.")
            print(f"Transcript saved to: {self.transcript_file}")


# ====================== MAIN ======================
if __name__ == "__main__":
    print("🔍 Detecting environment...")

    transcriber = HighGradeTranscriber()

    print("\n" + "=" * 70)
    print("CHOOSE MODE")
    print("=" * 70)
    print("1 = LIVE microphone (works on laptop/desktop)")
    print("2 = FILE upload mode (one file at a time - works everywhere)")
    print()

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        if IN_COLAB:
            print("❌ Live mode not possible in Colab (no microphone access).")
            print("Please choose option 2.")
        else:
            transcriber.start_live()

    elif choice == "2":
        print("\n" + "=" * 70)
        print("📤 FILE-BY-FILE MODE (upload one at a time)")
        print("• After each file you can upload the next one")
        print("• Press Cancel or close dialog to stop")
        print("=" * 70 + "\n")

        while True:
            print("\nUpload next audio file (or cancel to finish)...")
            try:
                if IN_COLAB:
                    uploaded = files.upload()
                else:
                    # Local fallback - ask for path
                    path = input(
                        "Enter full path to audio file (or press Enter to cancel): "
                    ).strip()
                    if not path:
                        break
                    uploaded = {os.path.basename(path): open(path, "rb").read()}

                if not uploaded:
                    print("\nUpload cancelled → stopping.")
                    break

                for fname in list(uploaded.keys()):
                    # Save to disk (required for librosa)
                    with open(fname, "wb") as f:
                        f.write(uploaded[fname])
                    transcriber.transcribe_file(fname)
                    # Optional: os.remove(fname)  # uncomment if you want to clean up

                print("\n✅ Done. Ready for next file ↓\n")

            except Exception as e:
                print(f"Error: {e}")
                break

        print(f"\nAll done! Combined transcript saved → {transcriber.transcript_file}")
        if IN_COLAB:
            files.download(transcriber.transcript_file)

    else:
        print("Invalid choice. Run the script again.")
