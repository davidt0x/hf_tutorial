#!/bin/bash

# Load the anaconda module. This gives us access to a modern version of python. We can
# use it to create a virtual environment that contains the specific version of python
# we want and the specific packages we need for our projects.
module load anaconda3/2022.5

# Create the anaconda virtual environment. You could also use standard python venv here
# if you prefer.
conda create -n hf python=3.9 -y

# Activate the environment
. activate hf

# Install dependencies for this tutorial, this line (without the transformers and
# datasets package) is taken directly from PyTorch documentation, see
# https://pytorch.org/get-started/locally/. This can also be installed with conda
# as well but there seems to be issues with torch 1.13 conda packages at the moment.
pip3 install torch torchvision torchaudio transformers[sentencepiece] datasets --extra-index-url https://download.pytorch.org/whl/cu116

# Also install the hugging face command line interface, useful for inspect your
# cache and deleting things.
pip3 install huggingface_hub[cli]
