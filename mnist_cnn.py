#for cnn
#syntax : nn.conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
#where in_channels is the number of input channels (e.g., 3 for RGB images),
# out_channels is the number of output channels (filters), 
# kernel_size is the size of the convolutional kernel, 
# stride is the step size for moving the kernel, 
# padding is the number of pixels added to the input on each side or in each edge. 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import  MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device',device)
transform = transforms.ToTensor()

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*7*7,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
def train_and_evaluate():  
    train_data = MNIST(
        root = 'data',
        train = True,
        transform = transform,
        download=True
    )
    test_data = MNIST(
        root = 'data',
        train = False,
        transform=transform,
        download=True
    )
    train_loader = DataLoader(
        train_data,
        batch_size = 32,
        shuffle = True
    )
    test_loader = DataLoader(
        test_data,
        batch_size = 32,
        shuffle = False
    )
    model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    cnn_losses = []

    for epoch in range(7):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(train_loader)
        cnn_losses.append(epoch_loss)
        print(f'Epochs {epoch+1}/7 | loss: {total_loss/len(train_loader):.4f}')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            predictions = output.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    torch.save(model.state_dict(), 'simple_cnn_model.pth')
    print('Model saved')
    return accuracy , cnn_losses
if __name__ == '__main__':
    accuracy , cnn_losses = train_and_evaluate()
    plt.plot(cnn_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss curve')
    plt.savefig('cnn_loss.png',dpi = 300 , bbox_inches='tight')
    plt.show()