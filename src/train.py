import torch
import torch.optim as optim
import torch.nn as nn
import src.utils
from src.evaluate import evaluate_model
from config import config as con

def train_model(model, train_loader, val_loader, con):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=con['learning_rate'])

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(con['num_epochs']):
        model.train()
        running_loss = 0.0
        for reactant_batch, product_batch in train_loader:
            reactant_batch, product_batch = reactant_batch.to(con['device']), product_batch.to(
                con['device'])
            optimizer.zero_grad()
            outputs = model(reactant_batch)
            loss = criterion(outputs, product_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}], Loss: {running_loss / len(train_loader)}")

        # Early stopping and validation logic (if applicable)
        if val_loader is not None:
            val_loss = evaluate_model(model, val_loader, con)
            print(f"Epoch [{epoch + 1}], Validation Loss: {val_loss}")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0  # Reset patience counter
                # Save the best model
                torch.save(model.state_dict(), r"C:\Users\amaan\PycharmProjects\OrganicChemistryReactionPredictor\venv\src\outputs\check_points\best_model.pth")
                print("Best model saved with validation loss: {:.4f}".format(val_loss))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= con['patience']:
                    print("Early stopping due to no improvement in validation loss.")
                    break
        # Save model checkpoint, etc.
