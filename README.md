# AudioTrain: Neural Amp Modeler & Trainer (AM3 / AM4 / IR Compatible)

AudioTrain is a modern, modular PyTorch neural amplifier training suite. It implements causal FIR-GRU-FIR neural architectures, perceptual loss formulations, and hardware-compatible binary serialization (`.am3Data`, `.am4Data`, `.irData`) for guitar amp hardware and VST plugins.

---

## Key Highlights

- **Exact Architectural Fidelity**:
  - **AM4**: 512-tap input FIR filter $\to$ GRU ($H=16$, $L=1$) $\to$ Linear ($16 \to 1$) $\to$ 512-tap output FIR filter.
  - **AM3**: Multi-channel CNN filter bank $\to$ 256-tap input FIR $\to \operatorname{arsinh}(x)$ saturation $\to$ GRU ($H=16$) $\to$ Linear $\to$ 256-tap output FIR.
  - **IR1024 / IR2048**: Linear impulse response convolution models (1024 or 2048 taps).
  - **NAM Importer / Distillation**: Full Neural Amp Modeler (`.nam` WaveNet) support, allowing distillation of third-party NAM models directly into lightweight AM3/AM4 hardware profiles.
- **Two-Phase Curriculum Training**:
  - **Phase 1: BasicTuning** ($Epoch < 80\%$): Pre-emphasized Time-Domain Error-to-Signal Ratio (ESR) with filter $y[n] = x[n] - 0.7225 x[n-2]$.
  - **Phase 2: FineTuning** ($Epoch \ge 80\%$): Pre-emphasized Spectral/Cepstral Loss with filter $H(z) = (1 - 0.85 z^{-1})^2$ and zero-phase FFT transform.
- **Automatic Re-amp Profile Detection (NAM vs. Legacy 60/40)**:
  - Automatically inspects Channel 0 (`dry`) of stereo WAV captures to detect whether the recording used a **NAM test signal** (`v3_0_0.wav`, `v2_0_0.wav`, `v1_1_1.wav`, or `Proteus_Capture.wav`) or a **legacy ANN / custom signal**, applying the optimal training and validation split for each.
- **Hardware-Ready Serialization**:
  - Produces byte-exact `.am3Data` and `.am4Data` binaries containing the 60-byte metadata/EQ header followed by contiguous float32 layer weights.
  - Exports `<model>.am4lossrec` with validation loss logs.
  - Exports rendered stereo `<model>_am4.wav` audio with peak loudness restored.
- **Dual Invocation Modes**:
  - Clean modern CLI with flag arguments (`--samples`, `--batch-size`, `--epochs`, `--model`, etc.).
  - Legacy backward-compatibility mode parsing the `?`-delimited string:
    `"<SamplesPath>?<BatchSize>?<Times>?<Delay>?<LRate>?<Model>"`

---

## Directory Structure

```
Sincomatt/
├── audiotrain/
│   ├── assets/                # Bundled re-amp & NAM distillation dry signals
│   │   ├── default_dry.wav    # Official NAM v3_0_0.wav (48 kHz, 24-bit mono)
│   │   └── v3_0_0_44100.wav   # Sinc-resampled NAM v3_0_0 (44.1 kHz, 24-bit mono)
│   ├── config.py              # Dataclass & YAML configuration loader
│   ├── configs/               # Preset configurations (AM3, AM4, IR, NAM)
│   │   ├── default.yaml
│   │   ├── am3.yaml
│   │   ├── am4.yaml
│   │   ├── ir1024.yaml
│   │   └── nam_to_am4.yaml
│   ├── data/
│   │   ├── audio.py           # 44.1 kHz sinc resampling & peak normalization
│   │   └── dataset.py         # 512-frame chunking & auto NAM / Legacy 60-40 split
│   ├── models/
│   │   ├── base.py            # BaseAmpModel interface
│   │   ├── am3.py             # AM3 neural model
│   │   ├── am4.py             # AM4 neural model
│   │   ├── ir.py              # Linear IR models
│   │   └── nam.py             # NAM model loader & distillation engine
│   ├── losses/
│   │   ├── preemphasis.py     # 2nd-order H(z) = (1 - 0.85 z^-1)^2 filter
│   │   ├── esr.py             # Time-domain ESR loss
│   │   ├── spectral.py        # Spectral ESR loss
│   │   └── curriculum.py      # Dual-phase curriculum controller
│   ├── serialization/
│   │   └── binary_format.py   # Byte-exact .am3Data / .am4Data packer & unpacker
│   ├── trainer.py             # Training engine & artifact exporter
│   └── cli.py                 # Unified CLI runner
├── train.py                   # Top-level training runner
├── main.py                    # Top-level main runner
├── requirements.txt
└── pyproject.toml
```

---

## Installation

Install dependencies using Python 3.9+:

```powershell
pip install -r requirements.txt
```

---

## Quick Start & Usage

### 1. Modern Command Line

Train an **AM4** model on a stereo WAV file:
```powershell
python train.py --samples samples/amp_sample.wav --model AM4 --epochs 1000 --batch-size 1024 --lr 0.0008
```

Train an **AM3** model:
```powershell
python train.py --samples samples/amp_sample.wav --model AM3 --epochs 1000
```

Train an **IR 1024** impulse response:
```powershell
python train.py --samples samples/amp_sample.wav --model IR1024 --epochs 500
```

Distill a **NAM model** (`.nam`) directly into an AM4 hardware profile:
```powershell
# Single NAM file distillation (uses built-in dry guitar excitation)
python train.py --samples models/my_amp.nam --model NAM_TO_AM4

# Batch convert an entire folder of .nam models into .am4Data hardware files
python train.py --samples models/ --model NAM_TO_AM4 --skip-existing

# (Optional) supply your own custom dry guitar excitation WAV
python train.py --samples models/my_amp.nam --dry-wav path/to/dry.wav
```

### 2. Using YAML Configuration Files

```powershell
python train.py --config audiotrain/configs/am4.yaml --samples samples/amp_sample.wav
```

You can customize parameters directly in the YAML file:
```yaml
samples_path: "samples/amp_sample.wav"
output_dir: "output"

data:
  sample_rate: 44100
  frame_size: 512
  delay: 0
  train_ratio: 0.6

model:
  model_type: "AM4"
  model_name: "my_custom_amp"
  fir1_size: 512
  fir2_size: 512
  rnn_hidden_size: 16

training:
  batch_size: 1024
  epochs: 1000
  learning_rate: 0.0008
  device: "cuda"
```

### 3. Batch Processing (Multi-Amp Queuing)

Train an entire directory, glob pattern, or list of captures with identical settings:

```powershell
# Train all WAV files in a directory
python train.py --samples path/to/amps/ --model AM4 --epochs 1000

# Train with wildcard / glob pattern
python train.py --samples "captures/*.wav" --model AM4 --epochs 1000

# Train multiple specific files
python train.py --samples amp_clean.wav amp_crunch.wav amp_lead.wav --model AM4

# Skip files that have already been trained (great for resuming interrupted batches)
python train.py --samples path/to/amps/ --model AM4 --skip-existing

# Recursively scan subfolders for captures
python train.py --samples path/to/amps/ --model AM4 --recursive

# Process from a text file list (one path per line)
python train.py --batch-file batch_queue.txt --model AM4
```

**Batch Features:**
- **Zero VRAM Leakage**: Automatic garbage collection and CUDA cache release (`torch.cuda.empty_cache()`) between models.
- **Fault Tolerance**: If one audio capture is corrupted, the runner logs the error, records `FAILED` in the ledger, and proceeds to the remaining files.
- **Summary Report**: Writes `batch_summary.json` containing timing, loss, and output paths, and prints an ASCII summary table.

### 4. Legacy String Invocation Format

If your automated pipeline or script passes a single `?`-delimited string argument, you can pass it directly:

```powershell
python train.py "samples\amp_sample.wav?1024?1000?0?0.0008?1"

# Directory paths also trigger batch processing in legacy mode:
python train.py "path\to\amps?1024?1000?0?0.0008?1"
```

**Parameter Order:**
1. `SamplesPath`: Path to the stereo WAV file, glob, or directory.
2. `BatchSize`: Frame count per batch (default `1024`).
3. `Times`: Total epochs (default `1000`).
4. `Delay`: Input lookahead delay in samples (default `0`).
5. `LRate`: AdamW learning rate (default `0.0008`).
6. `Model`: Model selector integer (`0`=AM3, `1`=AM4, `2`=IR1024, `3`=IR2048, `4`=NAM_TO_AM3, `5`=NAM_TO_AM4).

### 5. Running in Google Colab (Free T4 GPU)

A dedicated notebook [`AudioTrain_Colab.ipynb`](AudioTrain_Colab.ipynb) and pre-packaged archive are ready to go:
1. **Option A (Instant Zip Upload)**: Upload `audiotrain_colab.zip` directly into your Colab session. Cell 1 automatically unzips and installs everything in editable mode.
2. **Option B (Pip Package)**: You can also install the bundled distribution directly:
   ```bash
   !pip install dist/audiotrain-1.0.0.tar.gz
   ```
3. **Option C (Git Clone)**: Clone your repo directly into Colab:
   ```bash
   !git clone <your-repo-url>
   %cd Sincomatt
   !pip install -e .
   ```
4. Select a **GPU runtime** (`Runtime -> Change runtime type -> T4 GPU`).
5. Run the cells to train single models or batch folders (WAVs or `.nam` files), listen to previews, and download your hardware `.am4Data` / `.am3Data` binaries.

---

## Output Artifacts

By default, all generated output files are stored in the **same directory as the input audio file**. If you prefer a distinct folder, supply `--output-dir <path>`.

Upon completion (or when validation loss improves), the following files are saved:
- `<model_name>.am4Data` (or `.am3Data`, `.irData`): Raw binary model loadable by hardware/VST.
- `<model_name>.am4lossrec`: Plain-text file recording the best loss (`loss : 0.0947242`).
- `<model_name>.pt`: Standard PyTorch checkpoint containing model state dict and training history.
- `<model_name>_am4.wav` (optional, via `--render-audio`): Stereo WAV file containing original Dry on Ch0 and the rendered neural output on Ch1 for listening preview. (Disabled by default to maximize training and batch throughput).
- `batch_summary.json` (in batch mode): JSON summary of all models in the batch.

---

## Re-amping & Automatic Dataset Segmentation Modes

When training on a stereo WAV file (`Channel 0 = Dry Input`, `Channel 1 = Wet Amp Output`) or distilling a `.nam` profile, [`detect_reamp_profile`](audiotrain/data/dataset.py) automatically inspects Channel 0 using sample-rate-, bit-depth-, and gain-invariant acoustic signatures to select the matching train/validation segmentation strategy:

| Detected Profile | Source Input WAV | Total Duration | Skipped Intro | Training Split | Validation Split |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`NAM_V3`** | NAM `v3_0_0.wav` / `input.wav` (48k or 44.1k) | `~190.0s` (`3:10`) | `0:00 – 0:10` (`Val 1` + 1s silence) | **`0:10 – 3:01`** (`171.0s` / 95%) | **`3:01 – 3:10`** (`9.0s` guitar passage / 5%) |
| **`NAM_V2`** | NAM `v2_0_0.wav` | `~191.0s` (`3:11`) | None | **`0:00 – 2:51`** (`171.0s` / 95%) | **`2:51 – 3:00`** (`9.0s` guitar passage / 5%) |
| **`NAM_V1`** | NAM `v1_1_1.wav` / `v1.wav` | `~180.0s` (`3:00`) | None | **`0:00 – 2:51`** (`171.0s` / 95%) | **`2:51 – 3:00`** (`9.0s` guitar passage / 5%) |
| **`NAM_PROTEUS`** | GuitarML `Proteus_Capture.wav` | `~210.9s` (`3:30.9`) | `0:00 – 0:01` (blips) | **`0:01 – 3:20.9`** (`199.9s` / 95%) | **Final `10.0s`** (`3:20.9 – 3:30.9` / 5%) |
| **`LEGACY_60_40`** | Original ANN re-amps (`~59s`, `~173.7s`) or custom WAVs | Any | None | **First `60%`** of frames | **Final `40%`** of frames |

### Bundled Re-amp & Distillation Audio Files (`audiotrain/assets/`)

The bundled excitation signals in [`audiotrain/assets/`](audiotrain/assets/) are based on the standardized `v3_0_0.wav` (`input.wav`) test signal from [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) by Steven Atkinson, used under the MIT License:
- [`audiotrain/assets/default_dry.wav`](audiotrain/assets/default_dry.wav): Official **48 kHz, 24-bit mono** `v3_0_0.wav` signal (used automatically for `.nam` distillation at native 48 kHz and for 48 kHz DAW re-amping).
- [`audiotrain/assets/v3_0_0_44100.wav`](audiotrain/assets/v3_0_0_44100.wav): Sinc-resampled **44.1 kHz, 24-bit mono** `v3_0_0.wav` signal ready for direct 44.1 kHz DAW re-amping.

