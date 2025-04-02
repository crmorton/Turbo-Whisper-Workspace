"""
Setup script for Vocalis package
"""

from setuptools import setup, find_packages

setup(
    name="vocalis",
    version="0.1.0",
    description="Advanced audio processing, transcription, diarization, and security monitoring",
    author="8b.is",
    author_email="info@8b.is",
    url="https://vocal.is",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.1",
        "pydub>=0.25.1",
        "sherpa-onnx>=1.9.0",
        "huggingface-hub>=0.16.4",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "python-multipart>=0.0.6",
        "gradio>=3.40.0",
        "llama-cpp-python>=0.2.0; platform_system!='Windows'",  # Optional, not available on Windows
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.15.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)