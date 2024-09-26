import torch


def read_atoms(atoms):
    """
    Read ASE Atoms object and return a dictionary.

    Parameters:
    -----------
    atoms : ase.Atoms
        The ASE Atoms object.

    Returns:
    --------
    dict
        Dictionary with the following keys:
        - positions: torch.tensor
        - numbers: torch.tensor
        - pbc: torch.tensor
        - cell: torch.tensor
        - n_atoms: int
    """
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float64)
    numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int64)
    pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool)
    cell = torch.tensor(atoms.get_cell(), dtype=torch.float64)
    n_atoms = len(atoms)

    return {
        "positions": positions,
        "numbers": numbers,
        "pbc": pbc,
        "cell": cell,
        "n_atoms": n_atoms
    }
    

def pair_displacement(initial_atoms, final_atoms):
    """
    Pair displacement between two Atoms objects.

    Parameters:
    -----------
    initial_atoms : ase.Atoms
        Initial Atoms object.
    final_atoms : ase.Atoms
        Final Atoms object.
    device : str
        Device to use.

    Returns:
    --------
    torch.tensor
        Pair displacement.
    """
    pair = initial_atoms + final_atoms
    vec = [pair.get_distance(i, i + len(initial_atoms), mic=True, vector=True) for i in range(len(initial_atoms))]
    return torch.tensor(vec, dtype=torch.float64)