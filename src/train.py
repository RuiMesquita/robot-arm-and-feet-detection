import time
import torch

from torch.utils.data import DataLoader
from glob import glob
from utils import data
from model import build_unet
from utils import loss
from utils import functions


def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)
    return epoch_loss


if __name__ == "__main__":
    """ Ask for model name """
    model_name = input("Model name to be saved: ")

    """ Seeding """
    functions.seeding(42)

    """ Directories """
    functions.make_dir("target")

    """ Load dataset """
    train_x = sorted(glob("./data/train/images/*"))
    train_y = sorted(glob("./data/train/masks/*"))

    valid_x = sorted(glob("./data/test/images/*"))
    valid_y = sorted(glob("./data/test/masks/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 512
    W = 512
    size = (H, W)
    batch_size = 3
    num_epochs = 30
    lr = 1e-4
    checkpoint_path = f"./target/{model_name}.pth"

    """ Dataset and loader """
    train_dataset = data.DriveDataset(train_x, train_y)
    valid_dataset = data.DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device('cuda')
    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = loss.DiceBCELoss()

    """ Training the model """
    list_train_loss = []
    list_valid_loss = []
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f'Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}\n'

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        else:
            data_str = f"Valid loss not improved"

        end_time = time.time()
        epoch_mins, epoch_secs = functions.epoch_time(start_time, end_time)

        data_str += f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)

        """ Store data for the reports """
        functions.write_training_report(data_str)
        list_train_loss.append(train_loss)
        list_valid_loss.append(valid_loss)

    functions.generate_graph_report(num_epochs, list_train_loss, list_valid_loss)
