# %%
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %%
accelerator = torch.accelerator.current_accelerator()
device = accelerator.type if accelerator is not None else "cpu"
print(f"Using {device} device")

# %%
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Pad(2),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

val_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

# %%
batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


# %%
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 120, 5),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.layers(x)


class LeNet5BatchNorm(nn.Module):
    def __init__(self):
        super(LeNet5BatchNorm, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.BatchNorm2d(6),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 120, 5),
            nn.BatchNorm2d(120),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.layers(x)


# %%
lenet5 = LeNet5()
lenet5_batchnorm = LeNet5BatchNorm()

# %%
lr = 1e-3
optimizer = optim.SGD(lenet5.parameters(), lr=lr)
optimizer_batchnorm = optim.SGD(lenet5_batchnorm.parameters(), lr=lr)
lossfn = nn.CrossEntropyLoss()

# %%
epochs = 10


# %%
def train_epoch(model, dataloader, optimizer, lossfn, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = lossfn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / len(dataloader), correct / total


# %%
def validate(model, dataloader, lossfn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = lossfn(pred, y)

            total_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss / len(dataloader), correct / total


# %%
lenet5.to(device)
lenet5_batchnorm.to(device)

history = {
    "lenet5": {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []},
    "lenet5_bn": {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []},
}

print("Training LeNet5...")
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(
        lenet5, train_dataloader, optimizer, lossfn, device
    )
    val_loss, val_acc = validate(lenet5, val_dataloader, lossfn, device)

    history["lenet5"]["train_loss"].append(train_loss)
    history["lenet5"]["train_acc"].append(train_acc)
    history["lenet5"]["val_loss"].append(val_loss)
    history["lenet5"]["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

print("\nTraining LeNet5 with BatchNorm...")
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(
        lenet5_batchnorm, train_dataloader, optimizer_batchnorm, lossfn, device
    )
    val_loss, val_acc = validate(lenet5_batchnorm, val_dataloader, lossfn, device)

    history["lenet5_bn"]["train_loss"].append(train_loss)
    history["lenet5_bn"]["train_acc"].append(train_acc)
    history["lenet5_bn"]["val_loss"].append(val_loss)
    history["lenet5_bn"]["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )


# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(history["lenet5"]["train_loss"], label="LeNet5", marker="o")
axes[0, 0].plot(
    history["lenet5_bn"]["train_loss"], label="LeNet5 + BatchNorm", marker="s"
)
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Training Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history["lenet5"]["val_loss"], label="LeNet5", marker="o")
axes[0, 1].plot(
    history["lenet5_bn"]["val_loss"], label="LeNet5 + BatchNorm", marker="s"
)
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].set_title("Validation Loss")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history["lenet5"]["train_acc"], label="LeNet5", marker="o")
axes[1, 0].plot(
    history["lenet5_bn"]["train_acc"], label="LeNet5 + BatchNorm", marker="s"
)
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Accuracy")
axes[1, 0].set_title("Training Accuracy")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history["lenet5"]["val_acc"], label="LeNet5", marker="o")
axes[1, 1].plot(history["lenet5_bn"]["val_acc"], label="LeNet5 + BatchNorm", marker="s")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].set_title("Validation Accuracy")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lenet5_batchnorm_comparison.png")
plt.show()
