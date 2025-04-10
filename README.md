
# environment setup 
``` shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
# answer yes to terms and to automatically setting up Miniconda
# reopen terminal
conda deactivate
conda create -n e2e python=3.11
conda activate e2e
cd iwslt25_lowres
pip install -e .
```