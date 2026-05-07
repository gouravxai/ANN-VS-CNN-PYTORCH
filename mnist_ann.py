import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.ToTensor()
])
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
def train_and_evaluate():
    train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
    )
    test_data = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
    )
    train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=32,
    shuffle=False
    )
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ann_losses = []
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        ann_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/5 | Loss: {epoch_loss:.4f}")
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f" ANN Accuracy: {accuracy:.2f}%")
    return accuracy, ann_losses
if __name__ == '__main__':
    accuracy , ann_losses = train_and_evaluate()
    plt.plot(ann_losses)
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig('ann_loss.png',dpi=300,bbox_inches='tight')
    plt.show()