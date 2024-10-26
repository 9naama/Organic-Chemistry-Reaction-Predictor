import torch
from torch.utils.data import DataLoader
from torch import nn
from config import config as con

def evaluate_model(model, test_loader, con):
    model.eval()  # Set model to evaluation mode
    criterion = nn.MSELoss()
    test_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for reactant_batch, product_batch in test_loader:
            reactant_batch = reactant_batch.to(con['device'])
            product_batch = product_batch.to(con['device'])
            outputs = model(reactant_batch)
            loss = criterion(outputs, product_batch)
            test_loss += loss.item()

    average_loss = test_loss / len(test_loader)
    print(f"Test Loss (MSE): {average_loss}")
    return average_loss
