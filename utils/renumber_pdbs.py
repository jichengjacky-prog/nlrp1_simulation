#!/usr/bin/env python3
"""
Fix residue numbering in AF3-generated PDB files.
AF3 numbers all residues as '1'; this script renumbers them sequentially
(1, 2, 3, ...) using backbone N atoms as residue boundaries.
"""
import os
import glob
import shutil

PDB_DIR = "/home/ji.cheng4-umw/simulation_pipeline/targets/GPCRs/gpcr_agonist_human"
BACKUP_DIR = os.path.join(PDB_DIR, "_backup_before_renumber")

def renumber_pdb(filepath, backup=True):
    """Renumber residues sequentially in a PDB file. Returns stats dict."""
    if backup and not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

    basename = os.path.basename(filepath)
    if backup:
        shutil.copy2(filepath, os.path.join(BACKUP_DIR, basename))

    with open(filepath, 'r') as f:
        lines = f.readlines()

    res_num = 0
    first_n_seen = False
    atom_count = 0
    residue_count = 0
    updated_count = 0

    new_lines = []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line[12:16].strip()
            atom_count += 1

            if atom_name == "N":
                if not first_n_seen:
                    first_n_seen = True
                    res_num = 1
                else:
                    res_num += 1
                residue_count += 1

            # Only modify if residue number is different (all are "1" currently)
            old_resnum = line[22:26]
            new_resnum = f"{res_num:4d}"
            if old_resnum != new_resnum:
                updated_count += 1
                line = line[:22] + new_resnum + line[26:]

        new_lines.append(line)

    with open(filepath, 'w') as f:
        f.writelines(new_lines)

    return {
        'basename': basename,
        'atoms': atom_count,
        'residues': residue_count,
        'lines_updated': updated_count,
    }


def main():
    pdb_files = sorted(glob.glob(os.path.join(PDB_DIR, "*af3_replaced.pdb")))
    print(f"Found {len(pdb_files)} PDB files to renumber\n")

    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

    for pdb_path in pdb_files:
        stats = renumber_pdb(pdb_path, backup=True)
        print(f"  {stats['basename']}: {stats['residues']} residues, "
              f"{stats['atoms']} atoms, {stats['lines_updated']} lines updated")

    print(f"\nBackups saved to: {BACKUP_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
