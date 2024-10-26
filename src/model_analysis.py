import torch
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from src.DataProcessing import load_data
from src.model import ReactionPredictor
from config import config as con
import matplotlib.pyplot as plt


def evaluate_model_with_checks(model, test_loader, con):
    model.eval()  # Set the model to evaluation mode
    criterion = torch.nn.MSELoss()

    test_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for reactant_batch, product_batch in test_loader:
            print("Test batch product values:",
                  product_batch[:5].cpu().numpy())  # Print first 5 actual values
            break  # Just inspect the first batch for now
            reactant_batch = reactant_batch.to(con['device'])
            product_batch = product_batch.to(con['device'])

            # Forward pass to get predictions
            outputs = model(reactant_batch)

            # Check if the shape matches between predictions and actual values
            assert outputs.shape == product_batch.shape, f"Shape mismatch: {outputs.shape} vs {product_batch.shape}"

            # Compute the loss for the batch
            loss = criterion(outputs, product_batch)
            test_loss += loss.item()

            # Append predictions and actuals for further analysis
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(product_batch.cpu().numpy())

            # Print a few predictions and actuals to check
            print("Sample Predictions:", outputs[:5].cpu().detach().numpy())
            print("Sample Actuals:", product_batch[:5].cpu().detach().numpy())

    # Calculate the average test loss
    avg_test_loss = test_loss / len(test_loader)
    print(f"Average Test Loss: {avg_test_loss}")

    # Convert predictions and actuals to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Visualize predictions vs. actuals
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.show()

    return avg_test_loss, predictions, actuals

# Load data (only load the test set)
test_loader = load_data(con)  # Only need the test_loader

# Initialize the model architecture
model = ReactionPredictor(con['input_size'], con['hidden_size'], con['output_size'])
model = model.to(con['device'])

# Load the trained model's weights
model.load_state_dict(torch.load(r"C:\Users\amaan\PycharmProjects\OrganicChemistryReactionPredictor\venv\src\outputs\check_points\best_model.pth"))

avg_test_loss, predictions, actuals = evaluate_model_with_checks(model, test_loader, con)
# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(actuals, predictions)
print(f"Mean Absolute Error: {mae}")

# Calculate R² Score
r2 = r2_score(actuals, predictions)
print(f"R² Score: {r2}")
