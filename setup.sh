#!/bin/bash

conda create -n researchTopics python=3.8
conda activate researchTopics
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# conda install pytorch torchvision torchaudio cpuonly -c pytorch
# conda install -c huggingface transformers
# conda install -c huggingface -c conda-forge datasets
# conda install sentencepiece
# conda install gensim

conda install pytorch torchvision torchaudio cudatoolkit=11.3 transformers datasets sentencepiece gensim -c pytorch -c huggingface -c conda-forge jupyter