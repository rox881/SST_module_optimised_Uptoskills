# Speech-to-Text Optimization Pipeline

Optimized Faster-Whisper pipeline for improved accuracy, noise handling, and transcription clarity

## 📋 Project Overview

### Assigned Task
**Intern:** Gaurav Kshirsagar  
**Module:** Speech-to-Text Optimization  
**Task Brief:**  
Optimize the existing STT module for **better accuracy**, **improved noise handling**, and **transcription clarity**.

**Deliverables:**
- Improved STT pipeline (offline evaluation on dataset)
- Accuracy testing report with detailed metrics (WER, CER, insertions/deletions/substitutions)

This project addresses common STT challenges: hallucinations in noisy/silent segments, volume inconsistencies, and poor performance on spontaneous conversational speech.

### Project Goals & Achievements

| Goal                          | Achievement                                                                                     | Quantitative Outcome                          |
|-------------------------------|-------------------------------------------------------------------------------------------------|-----------------------------------------------|
| Better Accuracy               | Deterministic decoding + dynamic confidence filtering + model comparison                        | WER reduced from ~0.382 (Tiny baseline) to 0.2753 (Medium + VAD) — ~28% relative improvement |
| Improved Noise Handling       | Tuned VAD + A/B comparison (VAD on/off)                                                         | Insertions reduced by ~57% (hallucination proxy) |
| Transcription Clarity         | RMS normalization + long-audio chunking                                                         | Cleaner outputs, no clipping distortion, better handling of natural speech variations |

## 🗂️ Dataset Used

**Name:** Mozilla Spontaneous Speech Corpus – English  
**Link:** [https://datacollective.mozillafoundation.org/datasets/cmihqzerk023co20749miafhq](https://datacollective.mozillafoundation.org/datasets/cmihqzerk023co20749miafhq)

**Details:**
- ~130 MB `.tar.gz` archive
- Contains TSV metadata (`ss-corpus-en.tsv`) with transcriptions + MP3 audio clips
- Filtered to 150 validated clips (valid/test split) for fair, reproducible evaluation
- Represents real-world spontaneous, conversational English (disfluencies, pauses, varying volume, natural noise)

**Why this dataset?**  
It simulates authentic field conditions (non-scripted speech) — perfect for testing noise rejection and clarity without needing synthetic augmentation.

## 🛠️ Key Techniques Implemented

- **Preprocessing**  
  - Resampling to 16 kHz mono (Librosa)  
  - **RMS normalization** (replaced peak norm) — preserves dynamics, avoids clipping  

- **Long Audio Handling**  
  - Chunking into 25-second segments with 5-second overlap  

- **Inference Optimizations**  
  - Faster-Whisper (SYSTRAN optimized)  
  - Auto device selection: Medium + float16 (GPU) / Base + int8 (CPU)  
  - Deterministic: `temperature=0.0`, `beam_size=5`  
  - VAD: `threshold=0.4`, `min_silence_duration_ms=500`  

- **Hallucination Rejection**  
  - Dynamic log-probability threshold: stricter (-1.6) for long segments, lenient (-2.0) for short  
  - `no_speech_prob < 0.6` filter  

- **Evaluation**  
  - JiWER: full metrics (WER, CER, insertions, deletions, substitutions)  
  - 4 configurations tested: Base/Medium × VAD on/off  
  - Results aggregated and saved as CSV

## 🚀 How to Run

> **Three files are included in this project:**
> | File | Purpose | How to Run |
> |------|---------|------------|
> | `sst_uptoskill_optimisation_task.py` | Main evaluation pipeline (dataset benchmarking) | Run locally via Python |
> | `SST_UpTOSkill_Optimisation_Task.ipynb` | Notebook version of the evaluation pipeline | Run on Google Colab or Jupyter |
> | `high_grade_live_stt.py` | **Live microphone** + file-by-file transcription | Run locally via Python |

---

### 🔧 Common Setup (All Options)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rox881/SST_module_optimised_Uptoskills.git
   cd SST_module_optimised_Uptoskills
   ```

2. **Install dependencies** — choose **pip** or **conda**:

   **Using pip:**
   ```bash
   pip install -r requirements.txt
   ```

   **Using conda (recommended):**
   ```bash
   conda create -n stt_env python=3.10 -y
   conda activate stt_env
   pip install -r requirements.txt
   ```

   > ⚠️ On Linux, also install system audio/video drivers:
   > ```bash
   > sudo apt install -y ffmpeg libportaudio2
   > ```

---

### 🐍 Option 1: Evaluation Pipeline — `sst_uptoskill_optimisation_task.py`

**What it does:** Runs the full benchmark evaluation on the Mozilla Spontaneous Speech dataset (150 clips, 4 model configurations).

1. Place the dataset file in the project root:  
   `1764158905630-sps-corpus-1.0-2025-11-25-en.tar.gz`

2. Run:
   ```bash
   python sst_uptoskill_optimisation_task.py
   ```

**Output:**
- Console table with WER / CER / insertions / time per configuration
- `HIGH_GRADE_MODEL_COMPARISON.csv` saved in the current directory

---

### 📓 Option 2: Jupyter Notebook — `SST_UpTOSkill_Optimisation_Task.ipynb` (Colab / Jupyter)

**Setup on Google Colab:**

1. Upload the notebook file `SST_UpTOSkill_Optimisation_Task.ipynb` to Colab, **or** open it directly from GitHub.

2. Upload the dataset file in Colab:  
   `1764158905630-sps-corpus-1.0-2025-11-25-en.tar.gz`

3. Run **CELL 1** to install dependencies (run once):
   ```python
   !pip install -q faster-whisper jiwer pandas tqdm librosa torch torchaudio sounddevice
   !apt update -qq && apt install -y -qq ffmpeg libportaudio2
   ```

**Setup on Jupyter (Local):**

1. Install dependencies first:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Jupyter:
   ```bash
   jupyter notebook SST_UpTOSkill_Optimisation_Task.ipynb
   ```

**Run:**

- Execute **CELL 2** (main pipeline) — it will:
   - Auto-extract the `.tar.gz`
   - Process 150 clips
   - Test 4 configurations
   - Print comparison table
   - **Automatically download** `HIGH_GRADE_MODEL_COMPARISON.csv`

---

### 🎤 Option 3: Live Transcription — `high_grade_live_stt.py`

**What it does:** Real-time speech-to-text using your microphone **or** transcribe audio files one by one. Works offline on your local machine and also supports Google Colab (file mode only).

**Run:**
```bash
python high_grade_live_stt.py
```

You will be prompted to choose a mode:

| Mode | Description |
|------|-------------|
| **1 — Live Microphone** | Streams audio from your mic in real time and prints transcriptions as you speak. Press `Ctrl+C` to stop. *(Desktop/laptop only — not available in Colab)* |
| **2 — File Upload** | Transcribe audio files one at a time. In Colab it uses the upload dialog; locally it asks for the file path. |

**Output:**
- Live text printed to the console as you speak (mode 1) or after each file (mode 2)
- All transcriptions saved to `high_grade_transcript.txt`

## 📊 Example Results (150 clips)

| model  | vad   | avg_wer | avg_cer | avg_insertions | avg_time |
|--------|-------|---------|---------|----------------|----------|
| base   | True  | 0.2954  | 0.1523  | 1.2            | 0.58     |
| base   | False | 0.3121  | 0.1687  | 2.1            | 0.52     |
| medium | True  | 0.2753  | 0.1398  | 0.9            | 1.30     |
| medium | False | 0.2897  | 0.1452  | 1.5            | 1.25     |

**Recommended configuration:** **Medium with VAD** (best balance of accuracy and noise rejection)

## 🛠️ Challenges Faced & Solutions

| Challenge                              | Solution Implemented                                                                 |
|----------------------------------------|---------------------------------------------------------------------------------------|
| Volume variation & clipping            | Switched to RMS normalization (preserves dynamics, no distortion)                     |
| Hallucinations during silence/noise    | Tuned VAD + dynamic logprob threshold + no_speech_prob filter                         |
| Long audio files causing memory issues | 25s chunking with 5s overlap                                                          |
| TSV/MP3 path mismatches in archive     | Recursive glob search + fallback to secondary TSV file                                |
| Need for detailed diagnostics          | Full JiWER breakdown (WER, CER, insertions, deletions, substitutions)                 |

## 📄 License
MIT License

## 👤 Author
**Gaurav Kshirsagar**  
Intern – Speech-to-Text Optimization  
Email: gauravkshirsagar888@gmail.com  
LinkedIn: [https://linkedin.com/in/gaurav-kshirsagar-link/](https://linkedin.com/in/gaurav-kshirsagar-link/)  

**Last Updated:** March 10, 2026

---

Made with focus and dedication in Mumbai, India.
