import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from ase import Atoms
from ase.calculators.lj import LennardJones as LJ

from .base_class import PotentialBase

class LennardJones(PotentialBase):
    def __init__(self, numbers, epsilon=1.0, sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        if "minima" in kwargs:
            self.minima = kwargs["minima"]
        else:
            self.minima = None
        self.numbers = numbers
        self.atoms = Atoms(numbers=self.numbers)
        self.calculator = LJ(epsilon=epsilon, sigma=sigma)
        self.atoms.set_calculator(self.calculator)
    
    #@partial(jax.jit, static_argnums=(0,))
    def energy(self, positions):
        positions = self.point_transform(positions)
        # val = np.zeros(len(positions))
        val = [0.0 for _ in positions]
        for i, pos in enumerate(positions):
            self.atoms.set_positions(pos)
            val[i] = self.atoms.get_potential_energy()
        # atoms = Atoms(numbers=self.numbers, positions=positions)
        # val = atoms.get_potential_energy()
        # jax.experimental.io_callback(self.atoms.set_positions, None, positions, ordered=False)
        # val = self.atoms.get_potential_energy()
        return val
    
    def gradient(self, positions):
        positions = self.point_transform(positions)
        # val = np.zeros(len(positions))
        # grad = np.zeros((len(positions), len(self.numbers), 3))
        val = [0.0 for _ in positions]
        grad = [[[0.0 for _ in range(3)] for _ in self.numbers] for _ in positions]
        for i, pos in enumerate(positions):
            self.atoms.set_positions(pos)
            val[i] = self.atoms.get_potential_energy()
            grad[i] = self.atoms.get_forces()
        # atoms = Atoms(numbers=self.numbers, positions=positions)
        # val = atoms.get_potential_energy()
        # grad = atoms.get_forces()
        # jax.experimental.io_callback(self.atoms.set_positions, None, positions, ordered=False)
        # val = self.atoms.get_potential_energy()
        # grad = self.atoms.get_forces()
        return val, grad
