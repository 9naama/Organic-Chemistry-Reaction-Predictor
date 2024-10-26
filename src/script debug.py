import torch

# Load your saved .pt file
train_reactants = torch.load(r"C:\Users\amaan\PycharmProjects\OrganicChemistryReactionPredictor\venv\Data\processed\reactants_tensor.pt")

# Check the type of the data
print(f"Type of train_reactants: {type(train_reactants)}")

# If it's a list, print the type of elements in the list as well
if isinstance(train_reactants, list):
    print(f"Type of elements in train_reactants: {type(train_reactants[0])}")

# Do the same for train_products
train_products = torch.load(r"C:\Users\amaan\PycharmProjects\OrganicChemistryReactionPredictor\venv\Data\processed\products_tensor.pt")
print(f"Type of train_products: {type(train_products)}")

if isinstance(train_products, list):
    print(f"Type of elements in train_products: {type(train_products[0])}")
