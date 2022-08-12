import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


dataTrain = pd.read_csv('data/spaceship-titanic/train.csv')
dataTest = pd.read_csv('data/spaceship-titanic/test.csv')


def preprocess(df):
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df[['PGroup','PNr']] = df['PassengerId'].str.split('_', expand=True)
    df.drop(['Cabin', 'PassengerId', 'Name'], axis=1, inplace=True) # same as df.drop(columns=['Cabin', 'PassengerId'], inplace=True)

    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'Side', 'Deck'])

    df['Num'] = df['Num'].astype('float')
    df['PNr'] = df['PNr'].astype('int')
    df['PGroup'] = df['PGroup'].astype('float')
    

    df.fillna(df.median(axis=0), inplace=True)
    
    
    df_norm = (df - df.min()) / (df.max() - df.min())
    return df_norm



class dataset(Dataset):
    def __init__(self, xdata, ydata):
        self.labels = ydata
        self.inputs = xdata

    def __getitem__(self, index):
        
        return torch.tensor(self.inputs.iloc[index], dtype=torch.float), torch.tensor(self.labels.iloc[index], dtype=torch.float)

    def __len__(self):
        return len(self.inputs)

dataTrain['Transported'] = dataTrain['Transported'].astype('int')
lables = dataTrain.pop('Transported')

df_norm = preprocess(dataTrain)
train_dataset = dataset(df_norm, lables)

# df_norm, lables = preprocess(dataTest)
# test_dataset = dataset(df_norm, lables)

test_dataset = torch.tensor(preprocess(dataTest).to_numpy(), dtype=torch.float)


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred.squeeze(-1)



model = NeuralNetwork(df_norm.shape[1], 1)

learning_rate = 0.2
batch_size = 128
epochs = 10

loss_fn = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
print("Done!")


pred = model(test_dataset) > 0.5

submission = pd.read_csv('data/spaceship-titanic/test.csv')
submission["Transported"] = pred
submission[["PassengerId","Transported"]].to_csv('submission.csv', index=False)