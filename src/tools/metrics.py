import torch

class Metrics():

    def _parse_input(
            geo_val=None,
            velocity=None,
            pes_val=None,
            force=None,
            path=None,
            t=None,
            path_output=None,
            requires_velocity=False,
            requires_force=False,
            fxn_name=None
            ):
        inp_velocity = velocity is not None or not requires_velocity
        inp_force = force is not None or not requires_force
        use_input = geo_val is not None and pes_val is not None
        use_input = use_input and inp_velocity and inp_force
        if use_input:
            return geo_val, velocity, pes_val, force
        
        if path_output is not None and path is not None:
            raise ValueError("Cannot call metric functions with both path != None and path_output != None")
        
        if path_output is not None:
            pout_velocity = path_output.velocity is not None or not requires_velocity
            pout_force = path_output.force is not None or not requires_force
            if not pout_velocity or not pout_velocity:
                message = f"When calling {fxn_name} and providing path_output the "
                if not pout_velocity:
                    message += "velocity "
                    if not pout_force:
                        message += "and force "
                else:
                    message += "force "
                raise ValueError(message + "must be provided in the PathOutput.")
            return path_output.geometric_path, path_output.velocity,\
                path_output.potential_path, path_output.force
        
        if path is not None:
            if t is None:
                raise ValueError("Must specify evaluation times for path when using path argument")
            path_output = path(t, return_velocity=requires_velocity, return_force=requires_force)
            return path_output.geometric_path, path_output.velocity,\
                path_output.potential_path, path_output.force
        
        message = f"Cannot parse input arguments to {fxn_name}, please use one of the following opitons\n"
        message += f"\t1) Provide geometric_path and potential path, and if needed velocity and/or force\n"
        message += f"\t2) Provide a PathOutput class\n"
        message += f"\t3) Provide the path calculator and the time(s) to be evaluated"
        raise ValueError(message)
    
    def E_vre(self, **kwargs):
        kwargs['requires_force'] = True
        kwargs['requires_velocity'] = True
        kwargs['fxn_name'] = self.E_vre.__name__
        geo_val, velocity, pes_val, force = self._parse_input(**kwargs)
        
        return torch.linalg.norm(force)*torch.linalg.norm(velocity),\

    def E_pvre(self, **kwargs):
        kwargs['requires_force'] = True
        kwargs['requires_velocity'] = True
        kwargs['fxn_name'] = self.E_pvre.__name__
        geo_val, velocity, pes_val, force = self._parse_input(**kwargs)
        
        return torch.abs(torch.sum(velocity*force, dim=-1))

    def E_pvre_mag(self, **kwargs):
        kwargs['requires_force'] = True
        kwargs['requires_velocity'] = True
        kwargs['fxn_name'] = self.E_pvre.__name__
        geo_val, velocity, pes_val, force = self._parse_input(**kwargs)
        
        return torch.linalg.norm(velocity*force)#/jnp.linalg.norm(geo_grad)

    def vre(self, **kwargs):
        kwargs['requires_force'] = True
        kwargs['requires_velocity'] = True
        kwargs['fxn_name'] = self.E_pvre.__name__
        geo_val, velocity, pes_val, force = self._parse_input(**kwargs)
        
        e_pvre = self.E_pvre(
            geo_val=geo_val, velocity=velocity, pes_val=pes_val, force=force
        )
        e_vre = self.E_vre(
            geo_val=geo_val, velocity=velocity, pes_val=pes_val, force=force
        )
        return e_vre - e_pvre