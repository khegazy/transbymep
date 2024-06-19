import torch
from torch import nn

from .base_path import BasePath


class MLPnonreddistpath(BasePath):

    def __init__(
        self,
        potential,
        initial_point,
        final_point,
        n_embed=32,
        depth=3,
        seed=123,
        tol=1e-6,
        lr=1e-3,
        max_iter=1000,
    ):
        super().__init__(
            potential=potential,
            initial_point=initial_point,
            final_point=final_point,
        )
        self.ind = self.calc_nred(torch.stack([self.initial_point, self.final_point]), mode='min')
        self.initial_point_raw = self.initial_point
        self.initial_point = self.geo_to_dist(self.initial_point)
        self.final_point_raw = self.final_point
        self.final_point = self.geo_to_dist(self.final_point)
        self.activation = nn.SELU()
        input_sizes = [1] + [n_embed]*(depth - 1)
        output_sizes = input_sizes[1:] + [self.final_point.shape[-1]]
        self.layers = [
            nn.Linear(input_sizes[i//2], output_sizes[i//2]) if i%2 == 0\
            else self.activation\
            for i in range(depth*2 - 1)
        ]
        self.mlp = nn.Sequential(*self.layers)
        self.tol = tol
        self.lr = lr
        self.max_iter = max_iter

    def geometric_path(self, time, *args):
        scale = 1
        dist = scale * (self.mlp(time) - (1 - time) * self.mlp(torch.tensor([0.])) - time * self.mlp(torch.tensor([1.]))) \
            + ((1 - time) * self.initial_point + time * self.final_point)
        pos_trial = (1 - time) * self.initial_point_raw + time * self.final_point_raw
        return self.dist_to_geo(dist, pos_trial)
        
    def geo_to_dist(self, points):
        points = points.reshape(*points.shape[:-1], -1, 3)
        dist = torch.norm(points[..., self.ind[0], :] - points[..., self.ind[1], :], dim=-1)
        return dist
    
    # def dist_to_geo(self, dist_nred, pos_trial=None):
    #     if pos_trial is None:
    #         pos_trial = torch.randn_like(self.initial_point_raw)
    #     pos_trial = pos_trial.reshape(*pos_trial.shape[:-1], -1, 3)
    #     pos_trial.requires_grad = True
    #     optimizer = torch.optim.Adam([pos_trial], lr=self.lr)
    #     loss_fn = nn.MSELoss()
    #     for _ in range(self.max_iter):
    #         optimizer.zero_grad()
    #         dist_trial = torch.linalg.norm(pos_trial[..., self.ind[0], :] - pos_trial[..., self.ind[1], :], dim=-1)
    #         loss = loss_fn(dist_trial, dist_nred)
    #         if loss.item() < self.tol:
    #             break
    #         loss.backward(retain_graph=True)
    #         optimizer.step()
    #     else:
    #         # raise ValueError("Optimization did not converge")
    #         print("Optimization did not converge")
    #     return pos_trial.reshape(*pos_trial.shape[:-2], -1)

    def dist_to_geo(self, dist_nred, pos_ref=None):
        if pos_ref is None:
            pos_ref = (self.initial_point_raw + self.final_point_raw) / 2
        pos_ref = pos_ref.reshape(*pos_ref.shape[:-1], -1, 3)
        pos = torch.zeros_like(pos_ref)
        pos[..., self.ind[0, 0], :] = self.add_node()
        pos[..., self.ind[1, 0], :] = self.add_node(
            pos[..., self.ind[0, 0], :], dist_nred[..., 0],
            )
        pos[..., self.ind[1, 1], :] = self.add_node(
            pos[..., self.ind[0, 1], :], dist_nred[..., 1], 
            pos[..., self.ind[0, 2], :], dist_nred[..., 2],
            )
        for n in range(3, dist_nred.size(-1), 3):
            pos[..., self.ind[1, n], :] = self.add_node(
                pos[..., self.ind[0, n], :], dist_nred[..., n],
                pos[..., self.ind[0, n + 1], :], dist_nred[..., n + 1],
                pos[..., self.ind[0, n + 2], :], dist_nred[..., n + 2],
                sign=torch.sign(torch.det(pos_ref[..., [self.ind[1, n]], :] - pos_ref[..., self.ind[0, n:n+3], :]))
                )
        pos -= torch.mean(pos, dim=-2, keepdim=True)
        return pos.reshape(*pos.shape[:-2], -1)
        # raise NotImplementedError
        
    def calc_nred(self, points, mode='min'):
        points = points.reshape(*points.shape[:-1], -1, 3)
        n_atoms = points.size(-2)
        dist = torch.norm(points[..., :, None, :] - points[..., None, :, :], dim=-1)
        dist = dist.view(-1, *dist.shape[-2:]).min(dim=0).values    # shape (n_atoms, n_atoms)
        nodes_known = torch.zeros(n_atoms, dtype=torch.bool)
        ind = torch.triu_indices(n_atoms, n_atoms, 1)
        col, row = ind[:, dist[ind[0], ind[1]].argmin(keepdim=True)]
        ind_nred = torch.stack([row, col])
        nodes_known[col] = nodes_known[row] = True
        for n_nodes_known in range(2, n_atoms):
            dist_partial_sorted = dist[nodes_known, :][:, ~nodes_known].topk(min(3, n_nodes_known), largest=False, dim=0)    # shape (3, n_nodes_unknown)
            if mode == 'min':
                col_partial = dist_partial_sorted.values.min(dim=0).values    # shape (n_nodes_unknown)
                col_partial = col_partial.argmin()
            elif mode == 'max':
                col_partial = dist_partial_sorted.values.max(dim=0).values    # shape (n_nodes_unknown)
                col_partial = col_partial.argmin()
            elif mode == 'mean':
                col_partial = dist_partial_sorted.values.mean(dim=0)    # shape (n_nodes_unknown)
                col_partial = col_partial.argmin()
            else:
                raise ValueError(f'Unknown mode {mode}')
            col = torch.nonzero(~nodes_known)[col_partial]
            row_partial = dist_partial_sorted.indices[:, col_partial]
            row = torch.nonzero(nodes_known)[row_partial]
            ind_nred = torch.cat([ind_nred, torch.stack([row.flatten(), col.repeat(row.size(0))])], dim=1)    # shape (2, n_nodes_known)
            nodes_known[col] = True
        assert 3 * n_atoms - 6 <= ind_nred.size(1) <= 3 * n_atoms - 5, f'expected shape {3 * n_atoms - 6} or {3 * n_atoms - 5} but got {ind_nred.size(1)}'
        return ind_nred
    
    def add_node(self, point1=None, dist1=None, point2=None, dist2=None, point3=None, dist3=None, sign=1):
        if point1 is None:
            point = torch.zeros(3)
            return point
        elif point2 is None:
            point = point1.clone()
            point[..., 0] += dist1
            return point
        elif point3 is None:
            point = point1.clone()
            u = torch.norm(point2 - point1, dim=-1)
            x = (dist1 ** 2 - dist2 ** 2 + u ** 2) / (2 * u)
            y = torch.sqrt(dist1 ** 2 - x ** 2)
            point[..., 0] += x
            point[..., 1] += y
            return point
        else:
            point = point1.clone()
            xn = point2 - point1
            zn = torch.cross(xn, point3 - point1, dim=-1)
            yn = torch.cross(zn, xn, dim=-1)
            xn /= torch.norm(xn, dim=-1, keepdim=True)
            yn /= torch.norm(yn, dim=-1, keepdim=True)
            zn /= torch.norm(zn, dim=-1, keepdim=True)
            u = torch.sum((point2 - point1) * xn, dim=-1)
            vx = torch.sum((point3 - point1) * xn, dim=-1)
            vy = torch.sum((point3 - point1) * yn, dim=-1)
            x = (dist1 ** 2 - dist2 ** 2 + u ** 2) / (2 * u)
            y = (dist1 ** 2 - dist3 ** 2 + vx ** 2 + vy ** 2 - 2 * x * vx) / (2 * vy)
            z = torch.sqrt(torch.nn.functional.relu(dist1 ** 2 - x ** 2 - y ** 2, 0)) * sign
            point += x[..., None] * xn + y[..., None] * yn + z[..., None] * zn
            return point

        

    """
    def get_path(self, times=None):
        if times is None:
            times = torch.unsqueeze(torch.linspace(0, 1., 1000), -1)
        elif len(times.shape) == 1:
            times = torch.unsqueeze(times, -1)

        geo_path = self.geometric_path(times)
        pes_path = self.potential(geo_path)
        
        return geo_path, pes_path
    """
