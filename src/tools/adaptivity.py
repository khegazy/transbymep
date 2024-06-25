import torch


def _adaptively_add_y(ode_fxn, y, t, t_add, idxs_add):
    if len(t_add) == 0:
        RuntimeWarning("Do not expect empty points to add.")
        return y, t
    if y is None:
        y = torch.tensor([])
    if t is None:
        y = torch.tensor([])
    
    # Calculate new geometries
    #print("ADD PARALLEL EVAL TIMES", eval_times)
    y_add = ode_fxn(t_add)
    
    # Place new geometries between existing 
    y_combined = torch.zeros(
        (len(y)+len(y_add), y_add.shape[-1]),
        requires_grad=False
    ).detach()
    #print("GEO SHAPES", geos.shape, old_geos.shape, idxs_old.shape)
    y_idxs = None
    if y is not None and len(y):
        y_idxs = torch.arange(len(y), dtype=torch.int)
        bisected_idxs = idxs_add - torch.arange(len(idxs_add))
        for i, idx in enumerate(bisected_idxs[:-1]):
            y_idxs[idx:bisected_idxs[i+1]] += i + 1
        y_idxs[bisected_idxs[-1]:] += len(bisected_idxs)
        y_combined[y_idxs] = y
    assert(torch.all(y_combined[idxs_add] == 0))
    y_combined[idxs_add] = y_add

    # Place new times between existing 
    t_combined = torch.zeros(
        (len(y)+len(t_add), 1), requires_grad=False
    )
    #print("OLD IDXS", idxs_old[77:85])
    #print("NEW IDXS", idxs_new)
    if y_idxs is not None:
        t_combined[y_idxs] = t
    t_combined[idxs_add] = t_add

    return y_combined, t_combined


def _find_excess_y(p, error_ratios, remove_cut):
        
    #ratio_idxs_cut = torch.where(error_ratios < remove_cut)[0]
    ratio_mask_cut = error_ratios < remove_cut
    # Since error ratios encompasses 2 RK steps each neighboring element shares
    # a step, we cannot remove that same step twice and therefore remove the 
    # first in pair of steps that it appears in
    print("RC1", ratio_mask_cut)
    for idx in range(1, len(ratio_mask_cut)):
        ratio_mask_cut[idx] = ratio_mask_cut[idx] and not ratio_mask_cut[idx-1]
    print("RC2", ratio_mask_cut)
    ratio_idxs_cut = torch.where(ratio_mask_cut)[0]

    # Remove every other intermediate evaluation point
    ratio_idxs_cut = p*ratio_idxs_cut + 1
    ratio_idxs_cut = torch.flatten(
        ratio_idxs_cut.unsqueeze(1) + 2*torch.arange(p).unsqueeze(0)
    )
    y_mask_cut = torch.zeros((len(error_ratios)+1)*p+1, dtype=torch.bool)
    y_mask_cut[ratio_idxs_cut] = True

    return y_mask_cut
        
"""
    deltas = self._geo_deltas(geos)
    remove_mask = deltas < self.dxdx_remove
    #print("REMOVE DELTAS", deltas[:10])
    while torch.any(remove_mask):
        # Remove largest time point when geo_delta < dxdx_remove
        remove_mask = torch.concatenate(
            [
                torch.tensor([False]), # Always keep t_init
                remove_mask[:-2],
                torch.tensor([remove_mask[-1] or remove_mask[-2]]),
                torch.tensor([False]), # Always keep t_final
            ]
        )
        #print("N REMoVES", torch.sum(remove_mask), remove_mask[:10])

        #print("test not", remove_mask, ~remove_mask)
        eval_times = eval_times[~remove_mask]
        geos = geos[~remove_mask]
        deltas = self._geo_deltas(geos)
        remove_mask = deltas < self.dxdx_remove
    
    if len(eval_times) == 2:
        print("WARNING: dxdx is too large, all integration points have been removed")
    
    return geos, eval_times
"""


def __remove_idxs_to_ranges(idxs_cut):
    ranges_cut = []
    range_i = idxs_cut[0]
    idxC = 0
    while idxC < len(idxs_cut):
        for idxP in range(5):
            if idxs_cut[idxC+1] - idxs_cut[idxC] > 1:
                ranges_cut.append((range_i, idxs_cut[idxC]))
                range_i = idxs_cut[idxC+1]
        idxC += 1
    ranges_cut.append((range_i, idxs_cut[-1]))

    return ranges_cut

   
def _remove_idxs_to_ranges(idxs_cut):
    ranges_cut = []
    range_i = idxs_cut[0]
    idxC = 0
    while idxC < len(idxs_cut) - 1:
        if idxs_cut[idxC+1] - idxs_cut[idxC] > 1:
            ranges_cut.append((range_i, idxs_cut[idxC]))
            range_i = idxs_cut[idxC+1]
        idxC += 1
    ranges_cut.append((range_i, idxs_cut[-1]))

    return ranges_cut

def _find_sparse_y(t, p, error_ratios):
    print("SPARSE ERROR RATIO", error_ratios)
    ratio_idxs_cut = torch.where(error_ratios > 1.)[0]
    ratio_idxs_cut = p*ratio_idxs_cut + 1
    idxs_add = torch.flatten(
        ratio_idxs_cut.unsqueeze(1) + 2*torch.arange(p).unsqueeze(0)
    )

    t_add = (t[idxs_add-1] + t[idxs_add])/2
    idxs_add += torch.arange(len(idxs_add)) # Account for previosly added points
    
    return t_add, idxs_add


def _compute_error_ratios(y_p, y_p1, rtol, atol, norm):
    error_estimate = torch.abs(y_p1 - y_p)
    error_tol = atol + rtol*torch.max(y_p.abs(), y_p1.abs())
    error_ratio = norm(error_estimate/error_tol).abs() 

    error_estimate_2steps = error_estimate[:-1] + error_estimate[1:]
    error_tol_2steps = atol + rtol*torch.max(
        torch.stack(
            [y_p[:-1].abs(), y_p[1:].abs(), y_p1[:-1].abs(), y_p1[1:].abs()]
        ),
        dim=0
    )[0]
    error_ratio_2steps= norm(error_estimate_2steps/error_tol_2steps).abs() 
    
    return error_ratio, error_ratio_2steps