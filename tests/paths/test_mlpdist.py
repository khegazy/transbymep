import os
import pytest
import argparse
from ase.io import read
import numpy as np
import torch

from Popcornn import potentials, paths


def test_mlpdist(tmp_path, monkeypatch):
    # Create arguments and setup environment
    file_dir = os.path.dirname(os.path.abspath(__file__))

    # Test path
    potential = potentials.get_potential('lennard_jones')
    images = read(os.path.join(file_dir, '../configs/acetaldehyde.xyz'), index=':')
    path = paths.get_path(
        potential=potential,
        initial_point=images[0],
        final_point=images[-1],
        name='mlp_dist',
    )

    # Test shape
    for image, point in zip([images[0], images[-1]], [path.initial_point, path.final_point]):
        n_atoms = image.positions.shape[0]
        expected_shape = ((n_atoms + 4)) * ((n_atoms + 4) - 1) // 2
        shape = path.cart_to_dist(
            torch.tensor(image.get_positions(), device='cuda').reshape(1, -1)
        ).shape
        assert shape == (1, expected_shape)
        shape = point.shape
        assert shape == (expected_shape, )

    # Test distance
    for image, point in zip([images[0], images[-1]], [path.initial_point, path.final_point]):
        pos = image.get_positions()
        pos = np.concatenate([np.zeros((1, 3)), np.eye(3), pos], axis=0)
        ind = np.triu_indices(pos.shape[0], k=1)
        expected_dist = np.linalg.norm(pos[ind[0], :] - pos[ind[1], :], axis=-1)
        dist = path.cart_to_dist(
            torch.tensor(image.get_positions(), device='cuda').reshape(1, -1)
        ).detach().cpu().numpy()
        assert dist.flatten() == pytest.approx(expected_dist.flatten(), abs=1e-3)
        dist = point.detach().cpu().numpy()
        assert dist.flatten() == pytest.approx(expected_dist.flatten(), abs=1e-3)

    # Test geometry
    for image, point in zip([images[0], images[-1]], [path.initial_point, path.final_point]):
        expected_pos = image.get_positions()
        pos = path.dist_to_cart(point.reshape(1, -1)).detach().cpu().numpy().flatten()
        assert pos == pytest.approx(expected_pos.flatten(), abs=1e-3)

    # Test middle point
    expected_pos = np.mean([image.get_positions() for image in [images[0], images[-1]]], axis=0)
    # expected_dist = np.mean(
    #     np.linalg.norm(
    #         [image.get_positions()[:, None, :] - image.get_positions()[None, :, :] for image in [images[0], images[-1]]], axis=-1
    #     ), axis=0)
    pos = path(torch.tensor(0.5, device='cuda')).path_geometry.detach().cpu().numpy().squeeze(0)
    assert pos.shape == expected_pos.shape
    # dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    # assert dist == pytest.approx(expected_dist, abs=1e-1)