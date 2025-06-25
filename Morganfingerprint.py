from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def print_morgan_fingerprint_substructures(smiles, radius=2, nBits=2048):
    """
    Prints each bit set in the Morgan fingerprint and the corresponding substructure SMILES.
    
    Args:
        smiles (str): SMILES string of the molecule.
        radius (int): Radius for Morgan fingerprint (e.g., 2 = ECFP4).
        nBits (int): Size of the fingerprint bit vector.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES.")
        return

    bit_info = {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits, bitInfo=bit_info)

    print(f"SMILES: {smiles}")
    print(f"Number of bits turned on: {fp.GetNumOnBits()}\n")

    for bit, atom_infos in bit_info.items():
        print(f"Bit {bit} turned ON by:")
        for atom_idx, rad in atom_infos:
            env_bond_ids = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
            atoms = set()
            for bond_id in env_bond_ids:
                bond = mol.GetBondWithIdx(bond_id)
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())
            atoms.add(atom_idx)  # ensure central atom is included

            submol = Chem.PathToSubmol(mol, env_bond_ids)
            sub_smiles = Chem.MolToSmiles(submol)
            print(f"  - Atom index: {atom_idx}, Radius: {rad}, Substructure SMILES: {sub_smiles}")
        print()

print_morgan_fingerprint_substructures("C1CCCCCCC1")  # Phenol
