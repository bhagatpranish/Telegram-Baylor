#!/bin/bash

# Update package lists
sudo apt-get update -y

# Install Python 3, pip, and virtualenv
sudo apt-get install -y python3 python3-pip python3-venv

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install PyTorch with GPU support (if needed)
# Visit https://pytorch.org/get-started/locally/ to get the appropriate command for your system
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Install transformers, pandas, scikit-learn, keybert, PyPDF2, and googletrans
pip install transformers pandas scikit-learn keybert PyPDF2 googletrans==4.0.0-rc1

# Deactivate the virtual environment
deactivate
