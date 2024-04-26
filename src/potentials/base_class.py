import torch
from torch import nn

class PotentialBase(nn.Module):
    def __init__(self, add_azimuthal_dof=False, add_translation_dof=False, **kwargs) -> None:
        super().__init__()
        self.point_option = 0
        self.point_arg = 0
        if add_azimuthal_dof:
            self.point_option = 1
            self.point_arg = add_azimuthal_dof
        elif add_translation_dof:
            self.point_option = 2
        
        # Put model in eval mode
        self.eval()

    def point_transform(self, point, do_identity=False):
        if self.point_option == 0 or do_identity:
            return self.identity_transform(point)
        elif self.point_option == 1:
            return self.azimuthal_transform(point, self.point_arg)
        elif self.point_option == 2:
            return self.translation_transform(point)
    
    def identity_transform(self, points):
        return points

    def azimuthal_transform(self, points, shift):
        points = torch.transpose(points, 0, -1)
        points = torch.concatenate([
            torch.tensor([torch.sqrt(points[0]**2 + points[-1]**2)]) - shift,
            points[1:-1]
        ])
        return torch.tranpose(points, 0, -1)

    def translation_transform(self, point):
        point = torch.transpose(point, 0, -1)
        point = torch.concatenate([
            [point[0] + point[-1]],
            point[1:-1]
        ])
        return torch.transpose(point, 0, -1)