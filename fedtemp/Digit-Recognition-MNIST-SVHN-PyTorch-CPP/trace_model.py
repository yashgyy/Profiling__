import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main():
    # Create model instance
    model = Net()
    model.train()

    # Configure data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data_py', download = True, train=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.13707,), (0.3081,))
                      ])),
        batch_size=64, shuffle=True)

    # Configure optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(1, 11):  # 10 epochs
        batch_idx = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            batch_idx += 1
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}')
    
    # After training is complete, save the regular model
    # torch.save(model.state_dict(), 'net.pt')
    
    # Create traced model using actual MNIST data
    model.eval()  # Set to evaluation mode for tracing
    # Get one batch of MNIST data for tracing
    dataiter = iter(train_loader)
    trace_data, _ = next(dataiter)  # Get one batch
    trace_input = trace_data[0:1]  # Take just one image from the batch
    
    # Trace and save
    traced_script_module = torch.jit.trace(model, trace_input)
    traced_script_module.save('net_traced.pt')
    print("Traced model saved!")

if __name__ == '__main__':
    main()