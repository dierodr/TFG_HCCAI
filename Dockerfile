FROM python:3.12-slim

# Set working directory
WORKDIR /hccai

RUN mkdir -p /hccai/data/model_state/cirrhotic_state \
             /hccai/data/model_state/healthy_livers_or_not \
             /hccai/data/model_state/organ_classification \
             /hccai/data/csv

# Avoid interactive prompts and keep Python output clean
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# Install Python packages
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy application files
#App Files
COPY code/source/app /hccai/code/source/app
COPY code/source/config/ /hccai/code/source/config/
COPY code/source/utils/ /hccai/code/source/utils/

COPY code/source/ModelVisualizer.py /hccai/code/source/ModelVisualizer.py
COPY code/source/CustomCNN.py /hccai/code/source/CustomCNN.py
COPY code/source/PretrainedModel.py /hccai/code/source/PretrainedModel.py
COPY code/source/MetricsManager.py /hccai/code/source/MetricsManager.py

COPY code/__init__.py /hccai/code/__init__.py
COPY code/source/__init__.py /hccai/code/source/__init__.py

# Set Python eviroment
ENV PYTHONPATH=/hccai

CMD ["streamlit", "run", "/hccai/code/source/app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
#PYTHONPATH=$(pwd) streamlit run code/source/app/main.py --server.address 0.0.0.0 --server.port=8501 --server.headless true