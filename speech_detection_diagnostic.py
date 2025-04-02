#!/usr/bin/env python3
# speech_detection_diagnostic.py - A tool to diagnose speech detection issues
# Created by Hue and Aye - 8b.is Teamwork! 🎉

import os
import sys
import numpy as np
import soundfile as sf
import argparse
import logging
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

# Set up colorful logging
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
logger = logging.getLogger("speech_diagnostic")
logger.setLevel(logging.INFO)

# Create console handler with colored formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import numpy
        import soundfile
        import scipy
        import matplotlib
        logger.info("Required Python dependencies are installed!")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Installing required dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                  "numpy", "soundfile", "scipy", "matplotlib", "librosa"])
            logger.info("Dependencies installed successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False

def analyze_audio(input_file):
    """Analyze audio file for speech content"""
    try:
        import numpy as np
        import soundfile as sf
        import matplotlib.pyplot as plt
        import librosa
        
        logger.info(f"🎧 Analyzing audio file: {input_file}")
        
        # Load the audio file
        data, samplerate = sf.read(input_file)
        duration = len(data) / samplerate
        logger.info(f"Audio duration: {duration:.2f} seconds, Sample rate: {samplerate} Hz")
        
        # Convert to mono if stereo
        if len(data.shape) > 1 and data.shape[1] > 1:
            mono_data = np.mean(data, axis=1)
            logger.info(f"Converted stereo to mono for analysis")
        else:
            mono_data = data
        
        # Calculate basic statistics
        rms = np.sqrt(np.mean(mono_data**2))
        peak = np.max(np.abs(mono_data))
        dynamic_range = 20 * np.log10(peak / (np.mean(np.abs(mono_data)) + 1e-10))
        
        logger.info(f"Audio statistics:")
        logger.info(f"  - RMS level: {20 * np.log10(rms + 1e-10):.2f} dB")
        logger.info(f"  - Peak level: {20 * np.log10(peak + 1e-10):.2f} dB")
        logger.info(f"  - Dynamic range: {dynamic_range:.2f} dB")
        
        # Calculate spectral features
        stft = librosa.stft(mono_data)
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        # Calculate spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=mono_data, sr=samplerate)[0]
        logger.info(f"  - Mean spectral centroid: {np.mean(spectral_centroid):.2f} Hz")
        
        # Calculate zero crossing rate (high for noise, lower for speech)
        zcr = librosa.feature.zero_crossing_rate(mono_data)[0]
        logger.info(f"  - Mean zero crossing rate: {np.mean(zcr):.4f}")
        
        # Calculate energy entropy (measure of energy distribution)
        def energy_entropy(signal, frame_length=2048, hop_length=512):
            frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)
            energies = np.sum(frames**2, axis=0)
            energies = energies / np.sum(energies + 1e-10)
            entropy = -np.sum(energies * np.log2(energies + 1e-10))
            return entropy
        
        entropy = energy_entropy(mono_data)
        logger.info(f"  - Energy entropy: {entropy:.4f}")
        
        # Voice Activity Detection using librosa
        logger.info("Performing Voice Activity Detection...")
        
        # Method 1: Energy-based VAD
        energy = librosa.feature.rms(y=mono_data)[0]
        energy_threshold = np.mean(energy) * 1.5
        speech_frames_energy = energy > energy_threshold
        speech_percentage_energy = np.mean(speech_frames_energy) * 100
        logger.info(f"  - Energy-based VAD: {speech_percentage_energy:.2f}% speech detected")
        
        # Method 2: ZCR-based VAD (speech has lower ZCR than noise)
        zcr_threshold = np.mean(zcr) * 0.8
        speech_frames_zcr = zcr < zcr_threshold
        speech_percentage_zcr = np.mean(speech_frames_zcr) * 100
        logger.info(f"  - ZCR-based VAD: {speech_percentage_zcr:.2f}% speech detected")
        
        # Method 3: Combined VAD
        speech_frames_combined = np.logical_and(speech_frames_energy, speech_frames_zcr)
        speech_percentage_combined = np.mean(speech_frames_combined) * 100
        logger.info(f"  - Combined VAD: {speech_percentage_combined:.2f}% speech detected")
        
        # Create diagnostic plots
        plt.figure(figsize=(12, 10))
        
        # Plot waveform
        plt.subplot(3, 1, 1)
        plt.title('Waveform')
        librosa.display.waveshow(mono_data, sr=samplerate)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot spectrogram
        plt.subplot(3, 1, 2)
        plt.title('Spectrogram')
        librosa.display.specshow(stft_db, sr=samplerate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        
        # Plot VAD results
        plt.subplot(3, 1, 3)
        plt.title('Voice Activity Detection')
        frames = np.arange(len(speech_frames_energy))
        times = librosa.frames_to_time(frames, sr=samplerate)
        plt.plot(times, speech_frames_energy, label='Energy-based VAD')
        plt.plot(times, speech_frames_zcr, label='ZCR-based VAD')
        plt.plot(times, speech_frames_combined, label='Combined VAD')
        plt.xlabel('Time (s)')
        plt.ylabel('Speech Detected')
        plt.legend()
        
        # Save the plot
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, f"{os.path.basename(input_file)}_diagnostic.png")
        plt.tight_layout()
        plt.savefig(plot_file)
        logger.info(f"Diagnostic plot saved to: {plot_file}")
        
        # Provide recommendations
        logger.info("\n🔍 SPEECH DETECTION DIAGNOSIS:")
        
        if speech_percentage_combined < 10:
            logger.warning("⚠️ Very little speech detected in the audio file!")
            logger.info("Recommendations:")
            logger.info("1. Try processing without DeepFilterNet (--skip-deepfilter)")
            logger.info("2. If using DeepFilterNet, try a lower strength value (--deepfilter-strength 0.1)")
            logger.info("3. Check if the audio file actually contains speech")
        elif speech_percentage_combined < 30:
            logger.warning("⚠️ Low amount of speech detected in the audio file")
            logger.info("Recommendations:")
            logger.info("1. Try a lower DeepFilterNet strength (--deepfilter-strength 0.15)")
            logger.info("2. Use more aggressive Whisper settings (already implemented)")
        else:
            logger.info("✅ Good amount of speech detected in the audio file")
            logger.info("Recommendations:")
            logger.info("1. Current settings should work well")
            logger.info("2. If speech detection still fails, try --skip-deepfilter")
        
        # Return the diagnosis results
        return {
            "speech_percentage": speech_percentage_combined,
            "rms_db": 20 * np.log10(rms + 1e-10),
            "peak_db": 20 * np.log10(peak + 1e-10),
            "dynamic_range_db": dynamic_range,
            "spectral_centroid": np.mean(spectral_centroid),
            "zero_crossing_rate": np.mean(zcr),
            "energy_entropy": entropy,
            "plot_file": plot_file
        }
        
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def try_deepfilter_strengths(input_file, output_dir=None):
    """Try different DeepFilterNet strengths and analyze results"""
    try:
        # Check if DeepFilterNet is available
        try:
            from df import enhance, init_df
            deepfilter_available = True
        except ImportError:
            logger.warning("DeepFilterNet not available, attempting to install...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "deepfilternet"])
                from df import enhance, init_df
                deepfilter_available = True
                logger.info("🎉 DeepFilterNet installed successfully!")
            except Exception as e:
                logger.error(f"Failed to install DeepFilterNet: {e}")
                return None
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepfilter_test_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the audio file
        data, samplerate = sf.read(input_file)
        original_data = data.copy()
        
        # Check if we need to resample (DeepFilterNet works best with 48kHz)
        original_samplerate = samplerate
        if samplerate != 48000:
            try:
                import librosa
                logger.info(f"Resampling from {samplerate}Hz to 48000Hz for DeepFilterNet")
                data = librosa.resample(y=data.astype(np.float32), orig_sr=samplerate, target_sr=48000)
                samplerate = 48000
            except ImportError:
                logger.warning("Librosa not available for resampling. Will try to use DeepFilterNet anyway.")
        
        # Initialize DeepFilterNet model
        logger.info("Initializing DeepFilterNet model...")
        model, df_state, _ = init_df()
        
        # Process audio with DeepFilterNet
        logger.info("Applying noise suppression...")
        enhanced_data = enhance(model, df_state, data)
        
        # Resample back to original rate if needed
        if original_samplerate != 48000:
            try:
                import librosa
                logger.info(f"Resampling back to original rate: {original_samplerate}Hz")
                enhanced_data = librosa.resample(y=enhanced_data, orig_sr=48000, target_sr=original_samplerate)
                samplerate = original_samplerate
                # Also resample original data if it was resampled
                if original_data.shape != enhanced_data.shape:
                    original_data = librosa.resample(y=original_data.astype(np.float32), 
                                                    orig_sr=original_samplerate, 
                                                    target_sr=original_samplerate)
            except ImportError:
                pass
        
        # Try different strengths
        strengths = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        results = {}
        
        for strength in strengths:
            logger.info(f"\n🔍 Testing DeepFilterNet strength: {strength}")
            
            # Mix original and enhanced audio based on strength
            if strength == 0.0:
                mixed_data = original_data
            elif strength == 1.0:
                mixed_data = enhanced_data
            else:
                # Use adaptive mixing for intermediate strengths
                # Calculate a simple voice activity detection mask
                energy = np.abs(original_data)
                threshold = np.percentile(energy, 70)  # Use 70th percentile as threshold
                voice_mask = energy > threshold
                
                # Apply less filtering to high-energy regions (likely speech)
                speech_strength = strength * 0.5  # Use half strength for speech
                noise_strength = strength
                
                # Create a smoothed mask to avoid artifacts
                from scipy.ndimage import gaussian_filter1d
                if len(voice_mask.shape) > 1 and voice_mask.shape[1] > 1:
                    # Stereo
                    smooth_mask = np.zeros_like(voice_mask, dtype=float)
                    for ch in range(voice_mask.shape[1]):
                        smooth_mask[:, ch] = gaussian_filter1d(voice_mask[:, ch].astype(float), sigma=samplerate/100)
                else:
                    # Mono
                    smooth_mask = gaussian_filter1d(voice_mask.astype(float), sigma=samplerate/100)
                    
                # Apply adaptive mixing
                mixed_data = (1 - (smooth_mask * speech_strength + (1 - smooth_mask) * noise_strength)) * original_data + \
                            (smooth_mask * speech_strength + (1 - smooth_mask) * noise_strength) * enhanced_data
            
            # Save the mixed audio
            output_file = os.path.join(output_dir, f"deepfiltered_strength_{strength:.1f}_{os.path.basename(input_file)}")
            sf.write(output_file, mixed_data, samplerate)
            logger.info(f"Saved audio with strength {strength} to: {output_file}")
            
            # Analyze the mixed audio
            with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
                sf.write(temp_file.name, mixed_data, samplerate)
                temp_file.flush()
                analysis = analyze_audio(temp_file.name)
                if analysis:
                    results[strength] = analysis
                    logger.info(f"Speech percentage at strength {strength}: {analysis['speech_percentage']:.2f}%")
        
        # Find the best strength based on speech percentage
        if results:
            best_strength = max(results.items(), key=lambda x: x[1]['speech_percentage'])[0]
            logger.info(f"\n🏆 BEST DEEPFILTER STRENGTH: {best_strength}")
            logger.info(f"Speech percentage: {results[best_strength]['speech_percentage']:.2f}%")
            logger.info(f"Recommended command:")
            logger.info(f"python dynamic_bar_audio.py {input_file} --deepfilter-strength {best_strength}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error testing DeepFilterNet strengths: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Diagnose speech detection issues")
    parser.add_argument("input_file", help="Input audio file")
    parser.add_argument("--output-dir", "-o", help="Output directory for processed files")
    parser.add_argument("--test-deepfilter", "-d", action="store_true", 
                        help="Test different DeepFilterNet strengths")
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print(" 🎤 SPEECH DETECTION DIAGNOSTIC TOOL 🎤 ".center(80, "="))
    print(" Created by Hue and Aye - 8b.is Teamwork! ".center(80, "-"))
    print("=" * 80 + "\n")
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create output directory if needed
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze the audio file
    analysis = analyze_audio(args.input_file)
    
    # Test different DeepFilterNet strengths if requested
    if args.test_deepfilter:
        logger.info("\n🧪 Testing different DeepFilterNet strengths...")
        results = try_deepfilter_strengths(args.input_file, output_dir)
    
    logger.info("\n✅ Speech detection diagnostic complete!")
    logger.info("Check the diagnostic plots for more information.")

if __name__ == "__main__":
    main()
