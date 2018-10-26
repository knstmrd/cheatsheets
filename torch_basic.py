import torch
import torch.utils.data
import numpy as np


class DummyDataGenerator(torch.utils.data.Dataset):
    def __init__(self, feature_dim, train=True):
        self.feature_dim = feature_dim
        self.x_data = np.random.random((100, 100))
        self.train = train
        self.y_data = np.zeros((self.x_data.shape[0], 10))
        for i in range(10):
            self.y_data[:, i] = self.x_data.sum(axis=1) * i

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x_data[idx]).float(), torch.from_numpy(self.y_data[idx]).float()
        # return self.x_data[idx*self.batch_size:(idx + 1)*self.batch_size, :], self.y_data[idx*self.batch_size:(idx + 1)*self.batch_size, :]


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x


batch_size = 16
train_data = DummyDataGenerator(100)
test_data = DummyDataGenerator(100)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=True, num_workers=2)


torch_nn = TwoLayerNet(100, 50, 10)
print(torch_nn)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(torch_nn.parameters(), lr=0.0001, momentum=0.9, nesterov=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

n_epochs = 5

for epoch in range(n_epochs):

    torch_nn.train(True)
    for local_batch, local_labels in train_dataloader:

        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        optimizer.zero_grad()   # zero the gradient buffers
        output = torch_nn(local_batch)
        loss = criterion(output, local_labels)
        loss.backward()
        optimizer.step()

    torch_nn.eval()
    # Validation
    error = 0.0
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in test_dataloader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            output = torch_nn(local_batch)
            loss = criterion(output, local_labels)
            error += loss
    print('Epoch:{}, Error:{:.4f}'.format(epoch, error))  