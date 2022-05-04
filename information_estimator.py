import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from datasets.my_custom_dataset import MyCustomDataset

criterion = nn.MSELoss()

def train_rnn(data, labels):
    training_data = DataLoader(data, batch_size=5, shuffle=True)
    training_labels = DataLoader(labels, batch_size=5, shuffle=True)

    rnn = nn.RNN(25, 25, 3)

    optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01)

    lss_per_epoch = []
    for _ in range(1, 10):
        lss_per_batch = []
        for _, (batch_data, batch_labels) in enumerate(zip(training_data, training_labels)):
            optimizer.zero_grad()
            prob = rnn(batch_data)
            
            ## Take all the output states of the RNN, not the last hidden state
            loss = criterion(prob[0], batch_labels)
            lss_per_batch += [loss.item()]
            loss.backward()
            optimizer.step()
        lss_per_epoch += [sum(lss_per_batch) / len(lss_per_batch)]
    print(lss_per_epoch)
    return rnn

def test_rnn(rnn, data, labels):
    test_data = DataLoader(data, batch_size=10, shuffle=True)
    test_labels = DataLoader(labels, batch_size=10, shuffle=True)

    mse_per_batch = []
    for _, (batch_data, batch_labels) in enumerate(zip(test_data, test_labels)):
        prob = rnn(batch_data)
        mse = criterion(prob[0], batch_labels)
        mse_per_batch += [mse.item()]
    print('average mean squared error of batch: ')
    print(sum(mse_per_batch) / len(mse_per_batch))

print('information estimation of meansum: ')

yelp_path = 'D:/Users/EricvanSchaik/Downloads/processed_yelp_sample.json'
yelp_rnn = train_rnn(MyCustomDataset(path_to_dataset=yelp_path), MyCustomDataset(path_to_dataset=yelp_path, labels=True))

meansum_path = 'D:/Users/EricvanSchaik/Downloads/processed_summaries.json'
test_rnn(yelp_rnn, MyCustomDataset(path_to_dataset=meansum_path), MyCustomDataset(path_to_dataset=meansum_path, labels=True))

print('information estimation of pegasus: ')

xsum_path = 'D:/Users/EricvanSchaik/Downloads/xsum.json'
xsum_rnn = train_rnn(MyCustomDataset(path_to_dataset=xsum_path), MyCustomDataset(path_to_dataset=xsum_path, labels=True))

pegasus_path = 'D:/Users/EricvanSchaik/Downloads/pegasus_summaries.json'
test_rnn(xsum_rnn, MyCustomDataset(path_to_dataset=pegasus_path), MyCustomDataset(path_to_dataset=pegasus_path, labels=True))