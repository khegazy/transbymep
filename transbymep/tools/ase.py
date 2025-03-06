from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
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
    cell = torch.tensor(atoms.get_cell().array, dtype=torch.float64)
    n_atoms = len(atoms)
    tags = torch.tensor(atoms.get_tags(), dtype=torch.int64)

    return {
        "positions": positions,
        "numbers": numbers,
        "pbc": pbc,
        "cell": cell,
        "n_atoms": n_atoms,
        "tags": tags,
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
    vec = pair.get_distances(
        [i for i in range(len(initial_atoms))],
        [i + len(initial_atoms) for i in range(len(initial_atoms))],
        mic=True,
        vector=True,
    )
    return vec

def output_to_atoms(output, ref_images):
    """
    Convert output to ase.Atoms.
    
    Parameters:
    -----------
    output : paths.PathOutput
        Path output.
    ref_images : tools.Images
        Reference images.

    Returns:
    --------
    list[ase.Atoms]
        List of Atoms objects.
    """
    images = []
    for positions, energy, velocities, forces in zip(output.path_geometry, output.path_energy, output.path_velocity, output.path_force):
        atoms = Atoms(
            numbers=ref_images.numbers.detach().cpu().numpy(),
            positions=positions.detach().cpu().numpy().reshape(-1, 3),
            velocities=velocities.detach().cpu().numpy().reshape(-1, 3),
            pbc=ref_images.pbc.detach().cpu().numpy(),
            cell=ref_images.cell.detach().cpu().numpy(),
            tags=ref_images.tags.detach().cpu().numpy(),
        )
        calc = SinglePointCalculator(
            atoms,
            energy=energy.detach().cpu().numpy(),
            # forces=forces.detach().cpu().numpy().reshape(-1, 3),
        )
        atoms.calc = calc
        images.append(atoms)
    return images

def wrap_points(
        points: torch.Tensor,
        cell: torch.Tensor,
        center: float = 0
    ) -> torch.Tensor:
    """
    PyTorch implementation of ase.geometry.wrap_positions function.
    Assume periodic boundary conditions for all dimensions.
    """

    fractional = torch.linalg.solve(cell.T, points.view(*points.shape[:-1], -1, 3).transpose(-1, -2)).transpose(-1, -2)

    # fractional[..., :, self.pbc] %= 1.0
    fractional = (fractional + center) % 1.0 - center    # TODO: Modify this to handle partially true PBCs

    return torch.matmul(fractional, cell).view(*points.shape)