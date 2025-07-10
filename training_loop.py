import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import time
import os
from dataset_preparation import AudioDataset
from model import UNet1D


folder_path = r"/mnt/storage2/arobin/wave_u_net/train"
train_path = r"/mnt/storage2/arobin/wave_u_net/train"
sampling_Rate = 44100

# Define file extensions for order
file_extension = ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device set to {device}...')
##model
model = UNet1D(in_channels=1, out_channels=4)

# Move model to GPU and convert to float16
model.to(device) 
model.to(torch.float16)  # Convert weights to float16

# Ensure batch norm and layer norm layers remain in float32 (to avoid instability)
for module in model.modules():
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        module.to(torch.float32)  # Keep normalization layers in float32

# Convert model to use `float16` where necessary
for module in model.modules():
    if isinstance(module, nn.Conv1d) and module.bias is not None:
        module.bias = nn.Parameter(module.bias.to(dtype=torch.float16))  # Convert bias to float16

print('model initialization completed....')
# Dataset and Dataloader
# **Create Dataset and DataLoader**
batch_size = 2  # Adjust batch size based on RAM availability
dataset = AudioDataset(train_path, file_extension)
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)

print('initialize the dataloader.....')
# Loss and optimizer
##define a custom loss
def sdr_loss(y_pred, y_true, eps=1e-8):
    dot_product = torch.sum(y_pred * y_true, dim=-1, keepdim=True)
    true_norm = torch.sum(y_true**2, dim=-1, keepdim=True) + eps
    pred_norm = torch.sum(y_pred**2, dim=-1, keepdim=True) + eps
    return -torch.mean(dot_product / torch.sqrt(true_norm * pred_norm))

def combined_loss(y_pred, y_true, alpha=0.8):
    return alpha * nn.L1Loss()(y_pred, y_true) + (1 - alpha) * sdr_loss(y_pred, y_true)

criterion = combined_loss

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

# Mixed Precision
scaler = torch.amp.GradScaler()

# Logging
writer = SummaryWriter("runs/advanced_training")

# Training loop
epochs = 2
grad_accumulation_steps = 2  # Accumulate gradients every 2 batches
checkpoint_path = "model_checkpoint.pth"
print('Training started ......')

scaler = torch.cuda.amp.GradScaler()  # Initialize the GradScaler for mixed precision

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    start_time = time.time()

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Mixed Precision
        with torch.cuda.amp.autocast():  # Automatically use float16 for mixed precision
            outputs = model(inputs)
            loss = criterion(outputs, labels) / grad_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Update weights after accumulating gradients
        if (i + 1) % grad_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * grad_accumulation_steps
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    scheduler.step()

    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    elapsed_time = time.time() - start_time

    # Logging
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/train", epoch_acc, epoch)
    writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

    print(
        f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}% - Time: {elapsed_time:.2f}s"
    )

    # Save checkpoint
    torch.save(model.state_dict(), checkpoint_path)

writer.close()