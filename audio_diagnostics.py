#!/usr/bin/env python3
# audio_diagnostics.py - A tool to diagnose audio files for speech content
# Created by Hue and Aye - 8b.is Teamwork! 🎉

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from scipy.io import wavfile
import soundfile as sf
import librosa
import librosa.display

# Set up colorful logging
import logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[91m\033[1m', # Bold Red
        'ENDC': '\033[0m',     # Reset
    }

    def format(self, record):
        levelname = record.levelname
        message = super().format(record)
        return f"{self.COLORS.get(levelname, '')}{message}{self.COLORS['ENDC']}"

# Set up logger
logger = logging.getLogger("audio_diagnostics")
logger.setLevel(logging.INFO)

# Create console handler with colored formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def analyze_audio(audio_file, output_dir=None):
    """
    Analyze audio file for speech content
    
    Parameters:
    - audio_file: Path to the input audio file
    - output_dir: Directory to save analysis results (default: same as input)
    """
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return False
    
    try:
        # Create output directory if needed
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(audio_file))
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the audio file
        logger.info(f"Loading audio file: {audio_file}")
        data, samplerate = sf.read(audio_file)
        
        # Get basic audio information
        duration = len(data) / samplerate
        channels = 1 if len(data.shape) == 1 else data.shape[1]
        
        logger.info(f"Audio duration: {duration:.2f} seconds")
        logger.info(f"Sample rate: {samplerate} Hz")
        logger.info(f"Channels: {channels}")
        
        # Convert to mono if stereo
        if channels > 1:
            logger.info("Converting stereo to mono for analysis")
            data_mono = np.mean(data, axis=1)
        else:
            data_mono = data
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(data_mono**2))
        db_rms = 20 * np.log10(rms) if rms > 0 else -100
        logger.info(f"RMS energy: {db_rms:.2f} dB")
        
        # Calculate peak amplitude
        peak = np.max(np.abs(data_mono))
        db_peak = 20 * np.log10(peak) if peak > 0 else -100
        logger.info(f"Peak amplitude: {db_peak:.2f} dB")
        
        # Calculate signal-to-noise ratio (estimated)
        # We'll use the bottom 5% of frames as noise estimate
        frame_length = int(samplerate * 0.025)  # 25ms frames
        hop_length = int(samplerate * 0.010)    # 10ms hop
        
        # Compute frame-wise RMS
        frames = librosa.util.frame(data_mono, frame_length=frame_length, hop_length=hop_length)
        frame_rms = np.sqrt(np.mean(frames**2, axis=0))
        
        # Sort frames by energy
        sorted_rms = np.sort(frame_rms)
        noise_rms = np.mean(sorted_rms[:int(len(sorted_rms) * 0.05)])  # Bottom 5%
        signal_rms = np.mean(sorted_rms[int(len(sorted_rms) * 0.5):])  # Top 50%
        
        # Calculate SNR
        snr = 20 * np.log10(signal_rms / noise_rms) if noise_rms > 0 else 100
        logger.info(f"Estimated SNR: {snr:.2f} dB")
        
        # Detect speech presence using energy-based VAD
        speech_threshold = -40  # dB threshold for speech
        frames_above_threshold = np.sum(20 * np.log10(frame_rms + 1e-10) > speech_threshold)
        speech_percentage = frames_above_threshold / len(frame_rms) * 100
        
        if speech_percentage > 10:
            logger.info(f"✅ Speech detected! Approximately {speech_percentage:.1f}% of frames contain speech")
        else:
            logger.warning(f"⚠️ Low speech content detected: only {speech_percentage:.1f}% of frames above threshold")
        
        # Create plots
        plt.figure(figsize=(15, 10))
        
        # Plot waveform
        plt.subplot(3, 1, 1)
        plt.title("Waveform")
        time = np.linspace(0, duration, len(data_mono))
        plt.plot(time, data_mono)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        # Plot spectrogram
        plt.subplot(3, 1, 2)
        plt.title("Spectrogram")
        D = librosa.amplitude_to_db(np.abs(librosa.stft(data_mono)), ref=np.max)
        librosa.display.specshow(D, sr=samplerate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        
        # Plot energy over time
        plt.subplot(3, 1, 3)
        plt.title("Energy Over Time")
        frame_times = librosa.frames_to_time(np.arange(len(frame_rms)), sr=samplerate, hop_length=hop_length)
        plt.plot(frame_times, 20 * np.log10(frame_rms + 1e-10))
        plt.axhline(y=speech_threshold, color='r', linestyle='--', label=f'Speech Threshold ({speech_threshold} dB)')
        plt.xlabel("Time (s)")
        plt.ylabel("Energy (dB)")
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        plot_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_analysis.png")
        plt.tight_layout()
        plt.savefig(plot_file)
        logger.info(f"Analysis plot saved to: {plot_file}")
        
        # Generate a report
        report_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_audio_report.txt")
        with open(report_file, "w") as f:
            f.write("AUDIO DIAGNOSTIC REPORT\n")
            f.write("=====================\n\n")
            f.write(f"Audio File: {os.path.basename(audio_file)}\n")
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write(f"Sample Rate: {samplerate} Hz\n")
            f.write(f"Channels: {channels}\n\n")
            
            f.write("Audio Metrics:\n")
            f.write(f"- RMS Level: {db_rms:.2f} dB\n")
            f.write(f"- Peak Level: {db_peak:.2f} dB\n")
            f.write(f"- Estimated SNR: {snr:.2f} dB\n")
            f.write(f"- Speech Content: {speech_percentage:.1f}%\n\n")
            
            f.write("Analysis Results:\n")
            if speech_percentage > 10:
                f.write("✅ Speech detected in the audio file\n")
            else:
                f.write("⚠️ Low speech content detected\n")
                
            if snr < 10:
                f.write("⚠️ Low SNR - speech may be difficult to detect\n")
            
            if db_rms < -30:
                f.write("⚠️ Low audio level - consider normalizing\n")
                
            f.write("\nRecommendations:\n")
            if speech_percentage <= 10:
                f.write("- Try less aggressive noise suppression to preserve speech\n")
            if snr < 10:
                f.write("- Apply gentle noise reduction to improve SNR\n")
            if db_rms < -30:
                f.write("- Normalize audio to increase overall level\n")
        
        logger.info(f"Diagnostic report saved to: {report_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Audio Diagnostics Tool")
    parser.add_argument("input_file", help="Input audio file")
    parser.add_argument("--output-dir", "-o", help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print(" 🔍 AUDIO DIAGNOSTICS TOOL 🔍 ".center(80, "="))
    print(" Created by Hue and Aye - 8b.is Teamwork! ".center(80, "-"))
    print("=" * 80 + "\n")
    
    # Run analysis
    analyze_audio(args.input_file, args.output_dir)
    
    logger.info("Audio diagnostics complete!")

if __name__ == "__main__":
    main()
