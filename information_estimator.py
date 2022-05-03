import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from amazon_sample_dataset import AmazonSampleDataset
import numpy as np

summary_texts =  ['we need the information amount of this text', 'we need the information amount of this text']

def train_rnn(data, labels):
    training_data = DataLoader(data, batch_size=5, shuffle=True)
    training_labels = DataLoader(labels, batch_size=5, shuffle=True)

    rnn = nn.RNN(25, 25, 3)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01)

    lss_per_epoch = []
    for _ in range(1, 10):
        print('new epoch')
        lss_per_batch = []
        for _, (batch_data, batch_labels) in enumerate(zip(training_data, training_labels)):
            optimizer.zero_grad()
            prob = rnn(batch_data)
            loss = criterion(prob[0], batch_labels)
            lss_per_batch += [loss.item()]
            loss.backward()
            optimizer.step()
        lss_per_epoch += [sum(lss_per_batch) / len(lss_per_batch)]
    print(lss_per_epoch)

    return rnn

def test_rnn(rnn, summaries):
    return 0

train_rnn(AmazonSampleDataset(), AmazonSampleDataset(labels=True))
# test_rnn(summary_texts)