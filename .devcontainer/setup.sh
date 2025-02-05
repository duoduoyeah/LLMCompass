# Create a new conda environment with Python 3.9
conda create -n llmcompass_ae python=3.9 -y

# Activate the newly created environment
conda activate llmcompass_ae

# Install scalesim using pip
pip3 install scalesim

# Install PyTorch version 2.0.0 from the pytorch channel
conda install pytorch==2.0.0 -c pytorch -y

# Install additional Python packages using pip
pip3 install matplotlib seaborn scipy
