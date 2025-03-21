"""
Visualization utilities for CyberVox Audio Workspace
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib.figure import Figure

# Set cyberpunk theme for plots
try:
    plt.style.use('dark_background')
except Exception:
    # If dark_background style is not available, create our own dark theme
    plt.rcParams['figure.facecolor'] = '#121212'
    plt.rcParams['axes.facecolor'] = '#121212'
    plt.rcParams['text.color'] = '#ffffff'
    plt.rcParams['axes.labelcolor'] = '#ffffff'
    plt.rcParams['xtick.color'] = '#ffffff'
    plt.rcParams['ytick.color'] = '#ffffff'
    plt.rcParams['grid.color'] = '#333333'
    
CYBERPUNK_COLORS = ['#00ff9d', '#00ccff', '#ff00cc', '#ffcc00', '#cc00ff']

def plot_waveform(audio, sr, title="Waveform", figsize=(10, 4)):
    """
    Plot waveform of audio signal
    
    Args:
        audio: Audio signal as numpy array
        sr: Sample rate
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Calculate time axis
    times = np.arange(len(audio)) / float(sr)
    
    # Plot waveform
    ax.plot(times, audio, color=CYBERPUNK_COLORS[0], alpha=0.8)
    ax.fill_between(times, audio, alpha=0.2, color=CYBERPUNK_COLORS[0])
    
    # Add grid and style
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#111111')
    
    # Set labels and title
    ax.set_xlabel("Time (s)", color='#cccccc')
    ax.set_ylabel("Amplitude", color='#cccccc')
    ax.set_title(title, color=CYBERPUNK_COLORS[0], fontsize=14)
    
    # Add some styling
    fig.patch.set_facecolor('#111111')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(colors='#cccccc')
    
    fig.tight_layout()
    return fig

def plot_spectrogram(audio, sr, n_fft=2048, hop_length=512, title="Spectrogram", figsize=(10, 4)):
    """
    Plot spectrogram of audio signal
    
    Args:
        audio: Audio signal as numpy array
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for FFT
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)),
        ref=np.max
    )
    
    # Plot spectrogram
    img = librosa.display.specshow(
        D, 
        sr=sr, 
        hop_length=hop_length,
        x_axis='time',
        y_axis='log',
        ax=ax,
        cmap='viridis'
    )
    
    # Add colorbar
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='#cccccc')
    cbar.outline.set_edgecolor('#333333')
    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='#cccccc')
    
    # Add grid and style
    ax.grid(False)
    ax.set_facecolor('#111111')
    
    # Set labels and title
    ax.set_xlabel("Time (s)", color='#cccccc')
    ax.set_ylabel("Frequency (Hz)", color='#cccccc')
    ax.set_title(title, color=CYBERPUNK_COLORS[0], fontsize=14)
    
    # Add some styling
    fig.patch.set_facecolor('#111111')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(colors='#cccccc')
    
    fig.tight_layout()
    return fig

def plot_pitch_track(audio, sr, title="Pitch Track", figsize=(10, 4)):
    """
    Plot pitch track of audio signal
    
    Args:
        audio: Audio signal as numpy array
        sr: Sample rate
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Extract pitch using librosa
    pitches, magnitudes = librosa.piptrack(
        y=audio, 
        sr=sr,
        fmin=80,  # minimum frequency in Hz
        fmax=800  # maximum frequency in Hz
    )
    
    # Get pitches with highest magnitude at each time
    times = librosa.times_like(pitches)
    pitch_track = []
    
    for t, mag in zip(range(pitches.shape[1]), magnitudes.T):
        index = mag.argmax()
        pitch = pitches[index, t]
        if pitch > 0:  # Filter out zero pitches
            pitch_track.append((times[t], pitch))
    
    if pitch_track:
        t, p = zip(*pitch_track)
        # Plot pitch track
        ax.plot(t, p, color=CYBERPUNK_COLORS[1], marker='.', alpha=0.8, linestyle='-', markersize=4)
    
    # Add grid and style
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#111111')
    
    # Set labels and title
    ax.set_xlabel("Time (s)", color='#cccccc')
    ax.set_ylabel("Frequency (Hz)", color='#cccccc')
    ax.set_title(title, color=CYBERPUNK_COLORS[1], fontsize=14)
    
    # Add some styling
    fig.patch.set_facecolor('#111111')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(colors='#cccccc')
    
    fig.tight_layout()
    return fig

def plot_chromagram(audio, sr, n_fft=2048, hop_length=512, title="Chromagram", figsize=(10, 4)):
    """
    Plot chromagram (musical notes) of audio signal
    
    Args:
        audio: Audio signal as numpy array
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for FFT
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Compute chromagram
    chroma = librosa.feature.chroma_stft(
        y=audio, 
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Plot chromagram
    img = librosa.display.specshow(
        chroma, 
        sr=sr, 
        hop_length=hop_length,
        x_axis='time',
        y_axis='chroma',
        ax=ax,
        cmap='coolwarm'
    )
    
    # Add colorbar
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='#cccccc')
    cbar.outline.set_edgecolor('#333333')
    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='#cccccc')
    
    # Add grid and style
    ax.grid(False)
    ax.set_facecolor('#111111')
    
    # Set labels and title
    ax.set_xlabel("Time (s)", color='#cccccc')
    ax.set_ylabel("Pitch class", color='#cccccc')
    ax.set_title(title, color=CYBERPUNK_COLORS[2], fontsize=14)
    
    # Add some styling
    fig.patch.set_facecolor('#111111')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(colors='#cccccc')
    
    fig.tight_layout()
    return fig

def plot_speaker_diarization(segments, duration, title="Speaker Diarization", figsize=(10, 4)):
    """
    Plot speaker diarization segments
    
    Args:
        segments: List of dicts with speaker, start, and end times
        duration: Total duration of audio in seconds
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Group segments by speaker
    speakers = {}
    for segment in segments:
        speaker = segment['speaker']
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append((segment['start'], segment['end']))
    
    # Plot segments for each speaker
    y_pos = 0
    for i, (speaker, times) in enumerate(speakers.items()):
        color = CYBERPUNK_COLORS[i % len(CYBERPUNK_COLORS)]
        
        # Create a display label that doesn't start with underscore
        # (matplotlib ignores labels starting with underscore in legend)
        display_label = speaker
        if display_label.startswith('_'):
            display_label = display_label.lstrip('_')
        
        # Plot each segment for this speaker
        for start, end in times:
            ax.barh(
                y=y_pos,
                width=end-start,
                left=start,
                height=0.5,
                color=color,
                alpha=0.8,
                label=display_label if (start, end) == times[0] else ""
            )
        
        y_pos += 1
    
    # Add grid and style
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_facecolor('#111111')
    
    # Set labels and title
    ax.set_xlabel("Time (s)", color='#cccccc')
    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(list(speakers.keys()))
    ax.set_xlim(0, duration)
    ax.set_title(title, color=CYBERPUNK_COLORS[0], fontsize=14)
    
    # Add legend
    ax.legend(loc='upper right', facecolor='#111111', edgecolor='#333333', framealpha=0.9)
    
    # Add some styling
    fig.patch.set_facecolor('#111111')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(colors='#cccccc')
    
    fig.tight_layout()
    return fig
