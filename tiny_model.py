import torch
import torch.nn as nn

class TinyModel(nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = nn.Linear(1, 10)  # Adjusted for input size of 1 and hidden layer size of 10
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(10, 1)  # Adjusted for hidden layer size of 10 and output size of 1

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

tinymodel = TinyModel()

# Lets train this model with y=2x linear data
X = torch.randn(10000, 1)
y = 4 * X

# train
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(tinymodel.parameters(), lr=0.01)
for epoch in range(100):
    optimizer.zero_grad()
    output = tinymodel(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print('Epoch:', epoch, 'Loss:', loss.item())

# apply model on new data
X_new = torch.tensor([[0.1], [0.2], [0.3]])
y_pred = tinymodel(X_new)

print('Predictions:', y_pred)

# Save the model
torch.save(tinymodel.state_dict(), 'tinymodel.pth')
