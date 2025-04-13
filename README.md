
# Environment setup
install ffmpeg as torchaudio backend
``` shell
sudo apt update && sudo apt install ffmpeg
```
python environment set up
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

# Data
## Data Download
### Train
- [BigC](https://github.com/csikasote/bigc/tree/main)
- [FFSTC](https://huggingface.co/datasets/GbeBenin/FFSTC)
- [bem-eng synth](https://drive.google.com/file/d/1JIHLcP45fgPkFx9hizEGFfIam9Ha2zAB/view?usp=sharing)
- [fon-fra synth](https://drive.google.com/file/d/1AFkO-GUEqEkN0748eF6EYg_4JnbL-ydC/view?usp=sharing)

### Eval
- [BigC](https://github.com/csikasote/iwslt_2025_bem_eng_test)
- [FFSTC](https://huggingface.co/datasets/GbeBenin/testdata)

## Data prep
BigC
``` shell
sudo apt update && sudo apt install ffmpeg
```
