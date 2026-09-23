import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import HandwritingCNN


# ==========================================
# SETTINGS
# ==========================================

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5


# ==========================================
# DATA
# ==========================================

transform = transforms.ToTensor()


train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)


test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("Training examples:", len(train_dataset))
print("Test examples:", len(test_dataset))


# ==========================================
# MODEL
# ==========================================

model = HandwritingCNN()


# ==========================================
# LOSS
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# OPTIMISER
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# TRAINING
# ==========================================

print("\nStarting training...\n")


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0


    for images, labels in train_loader:

        # Clear gradients from previous batch
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Calculate gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        running_loss += loss.item()


    average_loss = (
        running_loss /
        len(train_loader)
    )


    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )


# ==========================================
# TEST MODEL
# ==========================================

print("\nTesting model...\n")


model.eval()


correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        predicted = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


accuracy = (
    100 *
    correct /
    total
)


print(
    f"Test accuracy: {accuracy:.2f}%"
)


# ==========================================
# SAVE MODEL
# ==========================================

torch.save(
    model.state_dict(),
    "handwriting_model.pth"
)


print(
    "\nModel saved as handwriting_model.pth"
)