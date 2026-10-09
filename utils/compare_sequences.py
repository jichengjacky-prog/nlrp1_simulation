#!/usr/bin/env python3
"""Compare sequences of af3_replaced.pdb files with corresponding best_cif files."""
import os
import re
import glob

pdb_dir = "/home/ji.cheng4-umw/simulation_pipeline/targets/GPCRs/gpcr_agonist_human"
cif_dir = os.path.expanduser("~/af3_projects/gpcr_agonist_human/best_cif")

# Standard 3-letter to 1-letter code
aa_3to1 = {
    'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F',
    'GLY':'G', 'HIS':'H', 'ILE':'I', 'LYS':'K', 'LEU':'L',
    'MET':'M', 'ASN':'N', 'PRO':'P', 'GLN':'Q', 'ARG':'R',
    'SER':'S', 'THR':'T', 'VAL':'V', 'TRP':'W', 'TYR':'Y',
    # Non-standard
    'HID':'H', 'HIE':'H', 'HIP':'H', 'HSD':'H', 'HSE':'H', 'HSP':'H',
    'CYX':'C', 'CYM':'C',
    'ASH':'D', 'GLH':'E',
    'LYN':'K',
    'ACE':'X', 'NME':'X',
    'MSE':'M',  # selenomethionine
    'SEP':'S',  # phosphoserine
    'TPO':'T',  # phosphothreonine
    'PTR':'Y',  # phosphotyrosine
}

def extract_seq_from_pdb(filepath):
    """Extract CA-based sequence from a PDB file.
    
    AF3-generated PDBs number all residues '1', so we collect CA atoms
    in file order (one CA per residue, no deduplication needed).
    Skips alternate conformers (altLoc not blank or 'A').
    """
    residues = []  # list of (chain, resname)
    with open(filepath) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                alt_loc = line[16]
                if alt_loc not in (' ', 'A'):
                    continue  # skip alternate conformer
                resname = line[17:20].strip()
                chain = line[21]
                residues.append((chain, resname))
    seq = ''.join(aa_3to1.get(r[1], 'X') for r in residues)
    return seq, residues

def extract_seq_from_cif(filepath):
    """Extract CA-based sequence from a CIF file.
    
    CIF ATOM_SITE columns (from _atom_site.* headers):
      group_PDB, id, type_symbol, label_atom_id, label_alt_id,
      label_comp_id, label_asym_id, label_entity_id, label_seq_id, ...
    We parse by splitting on whitespace and collect CA atoms sorted by
    (label_asym_id, label_seq_id).
    """
    ca_entries = []  # (seq_id, chain, resname)
    with open(filepath) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                if len(parts) < 9:
                    continue
                if parts[3] == "CA":   # label_atom_id
                    try:
                        seq_id = int(parts[8])   # label_seq_id
                    except ValueError:
                        seq_id = 0
                    chain = parts[6]             # label_asym_id
                    resname = parts[5]            # label_comp_id
                    ca_entries.append((seq_id, chain, resname))
    ca_entries.sort(key=lambda x: (x[1], x[0]))  # sort by chain then seq_id
    seq = ''.join(aa_3to1.get(r[2], 'X') for r in ca_entries)
    residues = [(ch, rn) for _, ch, rn in ca_entries]
    return seq, residues

def get_gene_uniprot_from_pdb(filepath):
    """Parse gene and UniProt from REMARK line."""
    with open(filepath) as f:
        for line in f:
            if "AF3-replaced structure:" in line:
                m = re.search(r'(\w+)\s+\((\w+)\)', line)
                if m:
                    return m.group(1), m.group(2)
    return None, None

# Find all PDB files
pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*af3_replaced.pdb")))
print(f"Found {len(pdb_files)} PDB files\n")

results = []

for pdb_path in pdb_files:
    basename = os.path.basename(pdb_path)
    gene, uniprot = get_gene_uniprot_from_pdb(pdb_path)

    if not gene or not uniprot:
        print(f"  SKIP: {basename} — Could not parse gene/uniprot from REMARK")
        continue

    expected_cif = f"{gene}_{uniprot}_best.cif"
    cif_path = os.path.join(cif_dir, expected_cif)

    if not os.path.exists(cif_path):
        # Try fuzzy match by gene name
        matches = glob.glob(os.path.join(cif_dir, f"{gene}_*_best.cif"))
        if len(matches) == 1:
            cif_path = matches[0]
        else:
            print(f"  SKIP: {basename} — CIF not found (tried {expected_cif})")
            continue

    pdb_seq, pdb_res = extract_seq_from_pdb(pdb_path)
    cif_seq, cif_res = extract_seq_from_cif(cif_path)

    match = (pdb_seq == cif_seq)
    results.append({
        'basename': basename,
        'gene': gene,
        'uniprot': uniprot,
        'cif_file': os.path.basename(cif_path),
        'pdb_len': len(pdb_seq),
        'cif_len': len(cif_seq),
        'match': match,
        'pdb_seq': pdb_seq,
        'cif_seq': cif_seq,
    })

    status = "MATCH" if match else "MISMATCH"
    print(f"  {status}: {basename} ({gene}/{uniprot}) — PDB={len(pdb_seq)}aa  CIF={len(cif_seq)}aa")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
matches = sum(1 for r in results if r['match'])
mismatches = sum(1 for r in results if not r['match'])
print(f"Total compared: {len(results)}  |  Matches: {matches}  |  Mismatches: {mismatches}")

if mismatches > 0:
    print("\n" + "=" * 60)
    print("MISMATCH DETAILS")
    print("=" * 60)
    for r in results:
        if not r['match']:
            print(f"\n--- {r['basename']} ({r['gene']}/{r['uniprot']}) vs {r['cif_file']} ---")
            print(f"  PDB length: {r['pdb_len']}  |  CIF length: {r['cif_len']}")
            # Find first difference
            pdb_s = r['pdb_seq']
            cif_s = r['cif_seq']
            common_len = min(len(pdb_s), len(cif_s))
            for i in range(common_len):
                if pdb_s[i] != cif_s[i]:
                    start = max(0, i - 5)
                    end = min(common_len, i + 6)
                    print(f"  First diff at residue {i+1}:")
                    print(f"    PDB[{start+1}:{end}]: {pdb_s[start:end]}")
                    print(f"    CIF[{start+1}:{end}]: {cif_s[start:end]}")
                    break
            else:
                # All common positions match, but lengths differ
                tail_pdb = pdb_s[common_len:]
                tail_cif = cif_s[common_len:]
                print(f"  Sequences identical up to residue {common_len}, then:")
                print(f"    PDB extra ({len(tail_pdb)}aa): {tail_pdb[:60]}")
                print(f"    CIF extra ({len(tail_cif)}aa): {tail_cif[:60]}")
