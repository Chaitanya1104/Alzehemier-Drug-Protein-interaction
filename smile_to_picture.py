from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Draw

def convert_smile_to_picture(smile: str) -> str:
    """
    Converts a smile string to a picture representation.
    Args:
        smile (str): The smile string to convert.
    """
    mol = MolFromSmiles(smile)
    print("Molecule object:", mol)
    image = Draw.MolToImage(mol)
    image.show()  # Display the image
    image.save("molecule_image.png")  # Save the image to a file
    print("Image saved as 'molecule_image.png'")

if __name__ == "__main__":
    smile=input("Enter a smile string:")
    convert_smile_to_picture(smile)