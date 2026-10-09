#!/usr/bin/env python3
"""Fetch SMILES for all ligands in the GPCR agonist input JSON.

Usage:
    python fetch_ligand_smiles.py                 # print SMILES only
    python fetch_ligand_smiles.py --update        # write SMILES back into the JSON

Sources, in order:
  - RCSB chemical component API
  - PDBe compound API
  - RCSB ideal SDF download
  - local MOL2 (RDKit)
  - local extracted ligand PDB (RDKit + bond perception)
"""
import json
import sys
import time
from pathlib import Path

import requests

BASE = Path('/home/ji.cheng4-umw/simulation_pipeline')
INPUT_JSON = BASE / 'targets' / 'jsons' / 'inputs_gpcr_agonist_human.json'


def fetch_smiles(resname: str):
    """Return a SMILES string for a PDB chemical component ID, or None."""
    # Primary: RCSB chemcomp REST API.
    try:
        r = requests.get(
            f'https://data.rcsb.org/rest/v1/core/chemcomp/{resname}',
            timeout=30,
        )
        if r.ok:
            data = r.json()
            if isinstance(data, dict):
                desc = data.get('rcsb_chem_comp_descriptor') or {}
                smi = desc.get('SMILES') or desc.get('smiles')
                if smi:
                    return smi.strip()
        else:
            print(f'  [debug] RCSB {resname} HTTP {r.status_code}: {r.text[:200]}', file=sys.stderr)
    except Exception as e:
        print(f'  [warn] RCSB chemcomp failed for {resname}: {e}', file=sys.stderr)

    # Fallback 1: PDBe chemical component API.
    try:
        r = requests.get(
            f'https://www.ebi.ac.uk/pdbe/api/pdb/compound/in_pdb/{resname}',
            timeout=30,
        )
        if r.ok:
            data = r.json()
            if isinstance(data, dict):
                for code, records in data.items():
                    if isinstance(records, list) and records:
                        smi = records[0].get('smiles')
                        if smi:
                            return smi.strip()
        else:
            print(f'  [debug] PDBe {resname} HTTP {r.status_code}: {r.text[:200]}', file=sys.stderr)
    except Exception as e:
        print(f'  [warn] PDBe fallback failed for {resname}: {e}', file=sys.stderr)

    # Fallback 2: RCSB SDF download endpoint.
    try:
        r = requests.get(
            f'https://files.rcsb.org/ligands/view/{resname}_ideal.sdf',
            timeout=30,
        )
        if r.ok:
            sdf_path = BASE / f'tmp_{resname}_ideal.sdf'
            sdf_path.write_bytes(r.content)
            try:
                from rdkit import Chem
                mol = Chem.MolFromMolFile(str(sdf_path), sanitize=False, removeHs=False)
                if mol is not None and mol.GetNumBonds() > 0:
                    return _rdmol_to_smiles(mol)
            finally:
                sdf_path.unlink(missing_ok=True)
        else:
            print(f'  [debug] RCSB SDF {resname} HTTP {r.status_code}: {r.text[:200]}', file=sys.stderr)
    except Exception as e:
        print(f'  [warn] RCSB SDF failed for {resname}: {e}', file=sys.stderr)

    return None


def _rdmol_to_smiles(mol):
    """Convert an RDKit mol to SMILES, tolerating kekulization failures."""
    from rdkit import Chem

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        pass
    if mol is None or mol.GetNumBonds() == 0:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        try:
            return Chem.MolToSmiles(mol, kekuleSmiles=False)
        except Exception:
            return None


def _local_smiles_from_mol2(resname, system_names):
    """Try to get SMILES from an existing local MOL2 using RDKit."""
    from rdkit import Chem

    for sysname in system_names:
        mol2 = BASE / 'toppar' / 'Ligands' / resname / f'{sysname}_{resname}.mol2'
        if not mol2.exists():
            continue
        try:
            mol = Chem.MolFromMol2File(str(mol2), sanitize=False, removeHs=False)
            if mol is not None and mol.GetNumBonds() > 0:
                return _rdmol_to_smiles(mol)
        except Exception as e:
            print(f'  [warn] local MOL2 SMILES failed for {resname}: {e}', file=sys.stderr)
    return None


def _atomic_num_from_atom_name(name):
    """Infer atomic number from a PDB atom name.

    Some AF3/cleaned ligand PDBs have 'A' in the element column for every
    atom, so RDKit cannot tell C/N/O/S apart.  The atom name prefix is a
    reliable fallback for these custom ligand files.
    """
    from rdkit import Chem

    name = (name or '').strip().upper()
    if not name:
        return 6  # default to carbon
    two_letter = {'CL': 17, 'BR': 35, 'NA': 11, 'MG': 12, 'FE': 26, 'ZN': 30}
    for prefix, num in two_letter.items():
        if name.startswith(prefix):
            return num
    one_letter = {'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'I': 53, 'H': 1}
    first = name[0]
    return one_letter.get(first, 6)


def _local_smiles_from_pdb(resname, system_names):
    """Try to get SMILES from an extracted ligand PDB using RDKit bond perception."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    for sysname in system_names:
        pdb = BASE / 'toppar' / 'Ligands' / resname / f'{sysname}_{resname}.pdb'
        if not pdb.exists():
            continue
        try:
            mol = Chem.MolFromPDBFile(str(pdb), sanitize=False, removeHs=False, proximityBonding=False)
            if mol is None:
                continue

            # Fix malformed element columns in AF3/cleaned ligand PDBs.
            for atom in mol.GetAtoms():
                info = atom.GetPDBResidueInfo()
                name = info.GetName().strip() if info is not None else ''
                atom.SetAtomicNum(_atomic_num_from_atom_name(name))

            if mol.GetNumBonds() == 0:
                rdDetermineBonds.DetermineConnectivity(mol)
            return _rdmol_to_smiles(mol)
        except Exception as e:
            print(f'  [warn] local PDB SMILES failed for {resname}: {e}', file=sys.stderr)
    return None


def _heavy_atom_count(smiles):
    """Return heavy-atom count for a SMILES string, or None if unparsable."""
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumHeavyAtoms() if mol is not None else None
    except Exception:
        return None


def main():
    update = '--update' in sys.argv

    data = json.loads(INPUT_JSON.read_text())

    # Unique resname -> list of system names that use it.
    ligand_resnames = {}
    for entry in data:
        for lig in entry.get('ligands', []):
            resname = lig['resname']
            ligand_resnames.setdefault(resname, []).append(entry['name'])

    print(f'Fetching SMILES for {len(ligand_resnames)} unique ligands...')
    fetched = {}
    for resname in sorted(ligand_resnames):
        systems = ligand_resnames[resname]

        api_smi = fetch_smiles(resname)
        local_mol2 = _local_smiles_from_mol2(resname, systems)
        local_pdb = _local_smiles_from_pdb(resname, systems)
        local_smi = local_mol2 or local_pdb

        if api_smi and local_smi:
            api_n = _heavy_atom_count(api_smi)
            local_n = _heavy_atom_count(local_smi)
            if api_n is not None and local_n is not None and abs(api_n - local_n) > 2:
                print(f'  [info] {resname}: API atom count {api_n} != local {local_n}; using local SMILES')
                smi = local_smi
                source = 'local-pdb' if local_pdb else 'local-mol2'
            else:
                smi = api_smi
                source = 'api'
        elif local_smi:
            smi = local_smi
            source = 'local-pdb' if local_pdb else 'local-mol2'
        else:
            smi = api_smi
            source = 'api'

        if smi:
            fetched[resname] = smi
            print(f'{resname}\t{smi}\t[{source}]')
        else:
            print(f'{resname}\tERROR', file=sys.stderr)
        time.sleep(0.2)

    if update:
        if not fetched:
            print('No SMILES were fetched; not updating the JSON.', file=sys.stderr)
            return 1

        for entry in data:
            for lig in entry.get('ligands', []):
                if lig['resname'] in fetched:
                    lig['SMILES'] = fetched[lig['resname']]

        INPUT_JSON.write_text(json.dumps(data, indent=2) + '\n')
        print(f'Updated {INPUT_JSON} with {len(fetched)} SMILES entries.')
    else:
        print(f'\nFetched {len(fetched)}/{len(ligand_resnames)} ligands.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
