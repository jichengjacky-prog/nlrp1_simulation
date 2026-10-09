"""Place a parameterized ligand using an explicit bound-pose atom mapping."""
import numpy as np


def align_named_coordinates(mobile_coords, mobile_names, reference_coords,
                            reference_names, atom_name_map):
    """Return all mobile coordinates fitted onto mapped reference atoms.

    Mapping keys are force-field/MOL2 names; values are source-PDB names.
    Fail on ambiguous or incomplete mappings rather than silently translating
    an arbitrary conformer into the binding pocket.
    """
    mobile = np.asarray(mobile_coords, dtype=float)
    reference = np.asarray(reference_coords, dtype=float)
    mobile_names, reference_names = list(mobile_names), list(reference_names)
    if mobile.shape != (len(mobile_names), 3) or reference.shape != (len(reference_names), 3):
        raise ValueError('Atom names and coordinate shapes do not match')
    if not np.isfinite(mobile).all() or not np.isfinite(reference).all():
        raise ValueError('Ligand alignment requires finite coordinates')
    if len(atom_name_map) < 3 or len(set(atom_name_map.values())) != len(atom_name_map):
        raise ValueError('At least three distinct mapped atoms are required')
    pairs = []
    for source, target in atom_name_map.items():
        if mobile_names.count(source) != 1 or reference_names.count(target) != 1:
            raise ValueError(f'Missing or ambiguous ligand atom mapping: {source} -> {target}')
        pairs.append((mobile_names.index(source), reference_names.index(target)))
    x = mobile[[i for i, _ in pairs]]
    y = reference[[j for _, j in pairs]]
    xc, yc = x.mean(axis=0), y.mean(axis=0)
    if min(np.linalg.matrix_rank(x-xc), np.linalg.matrix_rank(y-yc)) < 2:
        raise ValueError('Mapped ligand atoms must not be collinear')
    u, _, vt = np.linalg.svd((x-xc).T @ (y-yc))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    fitted = (mobile-xc) @ rotation+yc
    rmsd = np.sqrt(np.mean(np.sum((fitted[[i for i, _ in pairs]]-y)**2, axis=1)))
    return fitted, float(rmsd)
