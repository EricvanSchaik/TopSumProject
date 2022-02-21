import torch
from transformers import pipeline

use_cuda = True

if use_cuda and torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

print(pipeline('sentiment-analysis')('we love you'))
