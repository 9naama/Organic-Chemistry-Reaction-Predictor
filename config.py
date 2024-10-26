import torch
config = {
    'input_size': 2048,
    'hidden_size': 512,
    'output_size': 2048,
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'patience': 7,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # File paths
    'reactants_tensor_path': r"C:\Users\amaan\PycharmProjects\OrganicChemistryReactionPredictor\venv\Data\processed\reactants_tensor.pt",
    'products_tensor_path': r"C:\Users\amaan\PycharmProjects\OrganicChemistryReactionPredictor\venv\Data\processed\products_tensor.pt"
}
