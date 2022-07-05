#!/bin/bash

conda create -n researchTopics python=3.8
conda activate researchTopics
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch 
conda install transformers datasets -c huggingface -c conda-forge
conda install sentencepiece gensim jupyter bertopic