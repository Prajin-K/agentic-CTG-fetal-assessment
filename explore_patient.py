import sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Add repo to sys.path
REPO_ROOT = Path(r"c:\Users\prajin\Documents\CTG Challenge\ctg-agent-fetal-assessment-baseline")
sys.path.insert(0, str(REPO_ROOT))

from ctg_baseline.io import load_fetal_fhr
from ctg_baseline.signal_processing import DUS_filtering, create_scalogram_from_signal

def analyze_sample():
    data_dir = REPO_ROOT / "data"
    wav_path = data_dir / "task1/data/train/wav/t1_00001.wav"
    fetal_path = data_dir / "task1/labels/train/fetal/t1_00001.fetal"
    
    print(f"Loading: {wav_path}")
    fs_wav, raw_audio = wavfile.read(wav_path)
    if raw_audio.ndim > 1:
        raw_audio = raw_audio[:, 0]
    raw_audio = raw_audio.astype(np.float32)
    duration_sec = len(raw_audio) / fs_wav
    print(f"DUS Audio: Sample Rate = {fs_wav} Hz, Shape = {raw_audio.shape}, Duration = {duration_sec:.1f} s")
    print(f"DUS Stats: Min = {raw_audio.min():.1f}, Max = {raw_audio.max():.1f}, RMS = {np.sqrt(np.mean(raw_audio**2)):.2f}")
    
    print(f"\nLoading Ground Truth Reference: {fetal_path}")
    # Inspect raw header bytes
    with open(fetal_path, "rb") as f:
        header = f.read(8)
        version = header[0]
        channels = header[1]
        print(f"Header: Version = {version}, Channel Count = {channels}, Total File Size = {fetal_path.stat().st_size} bytes")
    
    fhr_reference = load_fetal_fhr(fetal_path)
    time_fhr = np.arange(len(fhr_reference)) / 4.0
    valid_mask = np.isfinite(fhr_reference)
    valid_fhr = fhr_reference[valid_mask]
    
    print(f"Reference FHR: Points = {len(fhr_reference)}, Duration = {time_fhr[-1]:.2f} s")
    print(f"Valid points = {len(valid_fhr)} / {len(fhr_reference)} ({len(valid_fhr)/len(fhr_reference)*100:.2f}%)")
    print(f"FHR Stats: Mean = {valid_fhr.mean():.1f} bpm, Median = {np.median(valid_fhr):.1f} bpm, Std = {valid_fhr.std():.2f} bpm")
    print(f"FHR Range: [{valid_fhr.min():.1f}, {valid_fhr.max():.1f}] bpm")
    
    # Process a 3.75s slice for scalogram
    slice_sec = 3.75
    slice_samples = int(slice_sec * fs_wav)
    slice_audio = raw_audio[fs_wav * 10 : fs_wav * 10 + slice_samples] # from sec 10 to 13.75
    filtered_slice = DUS_filtering(slice_audio, fs=fs_wav, lowcut=50.0, highcut=600.0)
    scalogram = create_scalogram_from_signal(filtered_slice, fs=fs_wav, time_bins=1000, freq_bins=40)
    
    # Create 4-panel visual plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), dpi=150)
    plt.subplots_adjust(hspace=0.4)
    
    # Panel 1: Full 20-minute FHR trace
    ax1 = axes[0]
    ax1.axhspan(110, 160, color='lightgreen', alpha=0.35, label='FIGO Normal Baseline (110 - 160 bpm)')
    ax1.plot(time_fhr, fhr_reference, color='#1B365D', lw=1.2, label='Patient Reference FHR (4 Hz)')
    ax1.set_xlim(0, 1200)
    ax1.set_ylim(80, 190)
    ax1.set_title("Panel 1: Full 20-Minute Clinical Fetal Heart Rate (t1_00001.fetal)", fontsize=12, fontweight='bold', pad=8)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Heart Rate (bpm)")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 2: 60-second zoom (sec 100 to 160)
    ax2 = axes[1]
    zoom_mask = (time_fhr >= 100) & (time_fhr <= 160)
    ax2.axhspan(110, 160, color='lightgreen', alpha=0.35)
    ax2.plot(time_fhr[zoom_mask], fhr_reference[zoom_mask], color='#D9381E', lw=2.0, marker='o', markersize=3, label='60-Second Detail (Beat-to-Beat Variability)')
    ax2.set_xlim(100, 160)
    ax2.set_ylim(120, 160)
    ax2.set_title("Panel 2: 60-Second Zoom — Observing Beat-to-Beat Variability (FIGO normal 5-25 bpm)", fontsize=11, fontweight='bold', pad=8)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("FHR (bpm)")
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 3: Raw DUS Audio slice (5 seconds: sec 10 to 15)
    ax3 = axes[2]
    time_audio = np.linspace(10, 15, int(5 * fs_wav))
    audio_5s = raw_audio[fs_wav * 10 : fs_wav * 15]
    ax3.plot(time_audio, audio_5s, color='#3B6978', lw=0.8, label='Raw 1 kHz Doppler Ultrasound Waveform')
    ax3.set_xlim(10, 15)
    ax3.set_title("Panel 3: Raw Doppler Ultrasound (DUS) Audio Waveform (5-Second Slice: 1,000 Hz)", fontsize=11, fontweight='bold', pad=8)
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Audio Amplitude")
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 4: CWT Morlet Scalogram
    ax4 = axes[3]
    im = ax4.imshow(scalogram, aspect='auto', origin='lower', extent=[10, 13.75, 1, 40], cmap='inferno')
    ax4.set_title("Panel 4: Morlet Continuous Wavelet Transform (CWT Scalogram: 3.75s window) — Heartbeat Energy Bursts", fontsize=11, fontweight='bold', pad=8)
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Wavelet Frequency Scale")
    fig.colorbar(im, ax=ax4, orientation='vertical', pad=0.015, label='Energy Intensity')
    
    out_img = Path(r"c:\Users\prajin\Documents\CTG Challenge\patient_t1_00001_analysis.png")
    plt.savefig(out_img, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved successfully to: {out_img}")

if __name__ == "__main__":
    analyze_sample()
