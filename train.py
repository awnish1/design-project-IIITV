import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Use CPU
device = torch.device("cpu")

# Load handwritten digit images
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=128,
    shuffle=True
)

# Define the neural network
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


model = DigitNet().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_fn = nn.CrossEntropyLoss()

# Train the network
for epoch in range(2):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}: "
        f"loss = {total_loss / len(train_loader):.4f}"
    )

# Save the trained weights
torch.save(model.state_dict(), "digit_model.pth")

print("Training complete!")
print("Weights saved to digit_model.pth")
