conda env create -f environment.yml
conda activate backdoordm
git clone https://github.com/quantum-bitss/diffusers.git
cd diffusers
pip install .
cd ..
