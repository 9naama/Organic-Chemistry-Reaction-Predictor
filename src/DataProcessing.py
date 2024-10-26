from torch.utils.data import TensorDataset, DataLoader
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import pandas as pd
import torch
from config import config as con
from sklearn.model_selection import train_test_split


def load_data(config):
    # # Load preprocessed full data
    # full_reactants = torch.load(r"C:\Users\amaan\PycharmProjects\OrganicChemistryReactionPredictor\venv\Data\processed\reactants_tensor.pt")
    full_products = torch.load(r"C:\Users\amaan\PycharmProjects\OrganicChemistryReactionPredictor\venv\Data\processed\products_tensor.pt")
    print("Sample test products:", full_products[:5])
    #
    # train_reactants, val_reactants = train_test_split(full_reactants, test_size=0.1)
    # train_products, val_products = train_test_split(full_products, test_size=0.1)
    # train_reactants = torch.stack([torch.tensor(fp) for fp in train_reactants])
    # val_reactants = torch.stack([torch.tensor(fp) for fp in val_reactants])
    # train_products = torch.stack([torch.tensor(fp) for fp in train_products])
    # val_products = torch.stack([torch.tensor(fp) for fp in val_products])
    # # Load preprocessed validation data (if applicable)

    # Load preprocessed test data
    test_reactants = torch.load(r"C:\Users\amaan\PycharmProjects\OrganicChemistryReactionPredictor\venv\Data\processed\reactants_tensor_test.pt")
    test_products = torch.load(r"C:\Users\amaan\PycharmProjects\OrganicChemistryReactionPredictor\venv\Data\processed\products_tensor_test.pt")
    test_reactants = torch.stack([torch.tensor(fp) for fp in test_reactants])
    test_products = torch.stack([torch.tensor(fp) for fp in test_products])
    # # Create DataLoader for training
    # train_dataset = TensorDataset(train_reactants, train_products)
    # train_loader = DataLoader(train_dataset, batch_size=con['batch_size'], shuffle=True)
    #
    # # Create DataLoader for validation
    # val_dataset = TensorDataset(val_reactants, val_products)
    # val_loader = DataLoader(val_dataset, batch_size=con['batch_size'], shuffle=False)

    # Create DataLoader for test
    test_dataset = TensorDataset(test_reactants, test_products)
    test_loader = DataLoader(test_dataset, batch_size=con['batch_size'], shuffle=False)

    # return train_loader, val_loader, test_loader
    return test_loader


def process_data(filePath):
    # Load the CSV file
    data = pd.read_csv(filePath, header=None)
    data.columns = ['Reactant', 'Product']
    # Lists to store valid fingerprints
    reactants_tensors = []
    products_tensors = []

    # Iterate over each row in the DataFrame to ensure pairs are processed together
    for _, row in data.iterrows():  # Process the first 10,000 rows as an example
        reactant_smiles = row['Reactant']
        product_smiles = row['Product']

        # Validate both reactant and product SMILES
        if is_valid_smiles(reactant_smiles) and is_valid_smiles(product_smiles):
            reactant_fp = smiles_to_fingerprint(reactant_smiles)
            product_fp = smiles_to_fingerprint(product_smiles)

            # Only add to lists if both fingerprints were successfully generated
            if reactant_fp is not None and product_fp is not None:
                reactants_tensors.append(reactant_fp)
                products_tensors.append(product_fp)
    torch.save(reactants_tensors, r"C:\Users\amaan\PycharmProjects\OrganicChemistryReactionPredictor\venv\Data\processed\reactants_tensor_test.pt")
    torch.save(products_tensors, r"C:\Users\amaan\PycharmProjects\OrganicChemistryReactionPredictor\venv\Data\processed\products_tensor_test.pt")
    # Final fingerprint arrays
    print(f"Number of valid reactant fingerprints: {len(reactants_tensors)}")
    print(f"Number of valid product fingerprints: {len(products_tensors)}")

# Define the function for converting SMILES to Morgan fingerprints
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Convert to Morgan fingerprint
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        rdk_bivector = generator.GetFingerprint(mol)
        np_fp = np.array(rdk_bivector, dtype=np.float32)
        return torch.from_numpy(np_fp)
    return None

def is_valid_smiles(smiles):
    """
    Check if a SMILES string can be converted and sanitized.
    Returns True if successful, False otherwise.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Attempt to sanitize and kekulize the molecule
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_KEKULIZE | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
            return True
    except:
        return False
    return False
