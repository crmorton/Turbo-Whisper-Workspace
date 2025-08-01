

git config --global user.email "7775975+crmorton@users.noreply.github.com"
git config --global user.name "crmorton"

apt install python3 python3-pip

# Install pyTorch cu118+cudnn8 (maximum of v2.3.1)
python3 -m pip install torch==2.3.1+cu118 numpy==1.26.3 torchvision torchaudio --index-url http://host.docker.internal:8081/repository/pypi-proxy-pytorch-cu118

# Install onnxruntime-gpu
# pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
pip install http://host.docker.internal:8081/repository/my-blob-store/aiinfra.pkgs.visualstudio.com/_packaging/pypi/download/onnxruntime-gpu/1.20.1/onnxruntime_gpu-1.20.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

# Install sherpa-onnx_cuda
wget https://k2-fsa.github.io/sherpa/onnx/cuda.html -O "sherpa-onnx_cuda.html"
sed -i 's|https://huggingface.co|http://host.docker.internal:8081/repository/huggingface-proxy|g' sherpa-onnx_cuda.html

apt install libasound2
python3 -m pip install sherpa-onnx==1.11.2+cuda -f sherpa-onnx_cuda.html
python3 -c "import sherpa_onnx; print(sherpa_onnx.__version__)"

# Install Web UI libraries
# Also installs: pydub pandas tqdm ruff huggingface-hub
python3 -m pip install gradio fastapi uvicorn

# Install Audio processing libraries
# Also installs: audioread scikit-learn scipy soundfile
apt install ffmpeg
python3 -m pip install librosa

# Install remaining libraries
python3 -m pip install transformers noisereduce matplotlib python-dotenv
