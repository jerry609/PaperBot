FROM e2bdev/code-interpreter:latest

# Base system dependencies for common paper reproduction workflows.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Core ML/scientific stack (CPU-only torch by default).
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    transformers datasets tokenizers accelerate safetensors \
    numpy scipy pandas matplotlib seaborn scikit-learn pillow \
    opencv-python-headless

# Developer tooling.
RUN pip install --no-cache-dir \
    pytest black pylint flake8 \
    jupyter ipykernel \
    pyyaml toml requests httpx tqdm

# Optional Node tooling for frontend reproduction tasks.
RUN npm install -g typescript ts-node

WORKDIR /home/user
