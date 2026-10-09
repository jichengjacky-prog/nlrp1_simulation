#!/usr/bin/env python3
"""
Pipeline Controller — Automate monitoring, build, equilibration, and production
for batch GPCR MD simulations.

Usage:
    python pipeline_controller.py status              # Print status of all targets
    python pipeline_controller.py run                 # Auto-advance all targets
    python pipeline_controller.py run --target NAME   # Advance specific target
    python pipeline_controller.py run --stage equil   # Only submit equilibration
    python pipeline_controller.py run --stage prod    # Only submit production
    python pipeline_controller.py run --dry-run       # Show what would be done
    python pipeline_controller.py retry-failed        # Retry failed jobs
    python pipeline_controller.py watch               # Monitor loop (Ctrl+C to stop)

Stages: build → equil → production (3 replicates)
"""

import os
import sys
import json
import time
import argparse
import subprocess
import shutil
import traceback
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────
BASEPATH = '/home/ji.cheng4-umw/simulation_pipeline/'
RESULTSPATH = BASEPATH + 'simulation_output/'
INPUT_JSON = BASEPATH + 'targets/jsons/inputs_gpcr_agonist_human.json'
ACEMD_PATH = '/home/ji.cheng4-umw/miniforge3/envs/GPCRmd/bin/acemd'
ACEMD_LICENSE = 'SG_LICENSE_FILE=28000@tolkien.prib.upf.edu,ACELLERA_LICENSE_SERVER=28000@tolkien.prib.upf.edu'
CONDA_PYTHON = '/home/ji.cheng4-umw/miniforge3/envs/GPCRmd/bin/python'
REPNUM = 3  # number of production replicates

# LSF settings
LSF_WALLTIME_EQUIL = 1440  
LSF_WALLTIME_PROD = 10080    
LSF_MEM = 4000
LSF_QUEUE = 'gpu'
LSF_NGPU = 1
LSF_NCPU = 2
MAX_GPU_JOBS = 16  # max running/pending GPU jobs before submission blocks


# ── Status constants ───────────────────────────────────────────────────────
class Stage:
    BUILD = 'build'
    EQUIL = 'equil'
    PROD = 'prod'

STATUS_PENDING = 'pending'
STATUS_RUNNING = 'running'
STATUS_DONE = 'done'
STATUS_FAILED = 'failed'
STATUS_PARTIAL = 'partial'
STATUS_SKIP = 'skip'


def is_simulation_done(logfile):
    """Return True if the ACEMD log shows successful completion."""
    if not os.path.exists(logfile):
        return False
    try:
        with open(logfile, 'r') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 3000))
            return '# Simulation completed!' in f.read()
    except Exception:
        return False


def _ensure_charmm_parameters(build_dir):
    """Recover the combined CHARMM parameter file for legacy build folders."""
    parameter_file = os.path.join(build_dir, 'parameters.prm')
    if os.path.exists(parameter_file) and os.path.getsize(parameter_file) > 0:
        return True

    topology_dir = os.path.join(build_dir, 'topologies')
    streams = sorted(str(path) for path in Path(topology_dir).glob('*.str'))
    if not streams:
        print(f"  ERROR: No CGenFF topology streams found in {topology_dir}")
        return False

    recovery_script = r'''
import glob
import os
import sys
sys.path.insert(0, __NOTEBOOKS_DIR__)
import charmm

build_dir = __BUILD_DIR__
output = os.path.join(build_dir, 'parameters.prm')
base = sorted(glob.glob(os.path.join(__TOPPAR_DIR__, '**', '*.prm'), recursive=True))
if not base:
    raise RuntimeError('No base CHARMM parameter files found')

charmm.combine(base, output)
for stream in __STREAMS__:
    charmm.split(stream, build_dir)

extras = sorted(
    path for path in glob.glob(os.path.join(build_dir, '*.prm'))
    if os.path.abspath(path) != os.path.abspath(output)
)
charmm.combine(extras + [output], output)
if not os.path.exists(output) or os.path.getsize(output) == 0:
    raise RuntimeError(f'Failed to create {output}')
print('PARAMETERS_READY')
'''
    recovery_script = (recovery_script
        .replace('__BUILD_DIR__', repr(build_dir))
        .replace('__NOTEBOOKS_DIR__', repr(os.path.join(BASEPATH, 'notebooks')))
        .replace('__TOPPAR_DIR__', repr(os.path.join(BASEPATH, 'toppar', 'TOP_PARAMS_ACE3')))
        .replace('__STREAMS__', repr(streams)))
    result = subprocess.run(
        [CONDA_PYTHON, '-c', recovery_script],
        capture_output=True, text=True, timeout=120, cwd=BASEPATH
    )
    if result.returncode != 0 or 'PARAMETERS_READY' not in result.stdout:
        print(f"  ERROR: Could not recover CHARMM parameters for {build_dir}")
        print(f"  {result.stderr[-1000:].strip()}")
        return False
    print(f"  Recovered CHARMM parameters: {parameter_file}")
    return True


def _write_acemd_runner(stage_dir):
    """Write a runner whose log stays with the job and whose exit code is ACEMD's."""
    run_sh = os.path.join(stage_dir, 'run.sh')
    with open(run_sh, 'w') as f:
        recovery = _read_recovery(stage_dir)
        command = (f'{CONDA_PYTHON} {BASEPATH}notebooks/resume_acemd.py'
                   if recovery.get('mode') == 'checkpoint' else ACEMD_PATH)
        f.write(f'#!/bin/bash\nexport WKFCOREDUMP_DISABLE=1\nexport ACEMD_NONINTERACTIVE=1 HTMD_NONINTERACTIVE=1 GPCRMD_OFFLINE=1\nexec {command} > log.txt 2>&1\n')
    os.chmod(run_sh, 0o755)



def _read_recovery(stage_dir):
    path = Path(stage_dir) / 'recovery.json'
    return json.loads(path.read_text()) if path.exists() else {}


def _prepare_production_recovery(stage_dir):
    """Preserve failed attempts, retaining usable production checkpoints."""
    root = Path(stage_dir)
    log = root / 'log.txt'
    if not log.exists() or 'Particle coordinate is NaN' not in log.read_text(errors='replace')[-20000:]:
        return
    if _read_recovery(stage_dir):
        raise RuntimeError(f'{stage_dir}: NaN recurred after recovery; inspect before another retry')
    checkpoint = (root / 'restart.chk').is_file() and (root / 'output.xtc').is_file()
    archive = root / ('failed_nan_' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    archive.mkdir()
    cfg = yaml.safe_load((root / 'input.yaml').read_text())
    if checkpoint and (cfg.get('barostat') or cfg.get('extforces')):
        raise ValueError('Checkpoint timestep recovery currently supports unrestrained NVT production only')
    for pattern in ('input.yaml', 'run.sh', 'job.sh', 'jobqueues.*', 'log.txt',
                    'restart.chk', 'output.csv', 'minimized.coor'):
        for old in root.glob(pattern):
            if checkpoint and old.name not in ('log.txt', 'jobqueues.failed', 'jobqueues.done'):
                shutil.copy2(old, archive / old.name)
            else:
                old.replace(archive / old.name)
    recovery = {'mode': 'checkpoint' if checkpoint else 'fresh',
                'archive': str(archive), 'timestep_fs': 1.0,
                'previous_timestep_fs': cfg.get('timestep', 4.0),
                'previous_trajectoryperiod': cfg.get('trajectoryperiod', 25000),
                'previous_hmr': cfg.get('hmr'),
                'previous_hydrogenmass': cfg.get('hydrogenmass', 4.032)}
    (root / 'recovery.json').write_text(json.dumps(recovery, indent=2) + '\n')
    print(f"  Preserved NaN attempt in {archive}; recovery mode: {recovery['mode']}")


def _apply_production_recovery(stage_dir):
    recovery = _read_recovery(stage_dir)
    if not recovery:
        return
    path = Path(stage_dir) / 'input.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['timestep'] = recovery['timestep_fs']
    cfg['restart'] = recovery['mode'] == 'checkpoint' or (
        Path(stage_dir, 'restart.chk').is_file() and Path(stage_dir, 'output.xtc').is_file())
    if recovery['mode'] == 'checkpoint':
        # ACEMD otherwise disables HMR automatically below 2 fs, changing masses.
        previous_hmr = recovery.get('previous_hmr')
        cfg['hmr'] = (recovery['previous_timestep_fs'] > 2
                      if previous_hmr is None else previous_hmr)
        cfg['hydrogenmass'] = recovery.get('previous_hydrogenmass', 4.032)
        # Keep the original physical frame spacing when appending to XTC.
        period = (recovery['previous_trajectoryperiod'] *
                  recovery['previous_timestep_fs'] / cfg['timestep'])
        if abs(period - round(period)) > 1e-6:
            raise ValueError('New timestep must divide the original trajectory interval')
        cfg['trajectoryperiod'] = int(round(period))
    else:
        cfg['minimize'] = max(10000, int(cfg.get('minimize', 0)))
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


# ── Helper: load input targets ────────────────────────────────────────────

def _configure_charmm_inputs(stage_dir, source_dir):
    """Use the merged parameters and retain custom topology atom types.

    ACEMD's protocol auto-detects every ``*.prm`` in the source directory.
    Split ligand parameter files can then be parsed before ``parameters.prm``;
    GAFF atom types are declared in the matching topology file, so OpenMM
    rejects the first nonbonded term as an unknown type. Load only the merged
    parameter file and prepend the generated topologies that declare custom
    atom types.
    """
    input_yaml = os.path.join(stage_dir, 'input.yaml')
    merged_parameters = os.path.join(stage_dir, 'parameters.prm')
    if not os.path.exists(input_yaml):
        raise FileNotFoundError(f'Missing ACEMD input file: {input_yaml}')
    if not os.path.exists(merged_parameters):
        raise FileNotFoundError(
            f'Missing merged CHARMM parameters: {merged_parameters}'
        )

    topology_names = []
    for source_path in sorted(Path(source_dir).glob('top_*_toppar.rtf')):
        destination = os.path.join(stage_dir, source_path.name)
        if os.path.abspath(str(source_path)) != os.path.abspath(destination):
            shutil.copy2(source_path, destination)
        topology_names.append(source_path.name)

    with open(input_yaml, 'r') as fy:
        cfg = yaml.safe_load(fy) or {}
    cfg['parameters'] = topology_names + ['parameters.prm']
    with open(input_yaml, 'w') as fy:
        yaml.safe_dump(cfg, fy, sort_keys=False)

    if topology_names:
        print(
            f"  Added {len(topology_names)} custom topology file(s) to "
            f"{stage_dir}/input.yaml"
        )

def load_targets():
    with open(INPUT_JSON) as f:
        return json.load(f)


def get_sysname(entry):
    """Return the system directory name (with optional _apo suffix)."""
    name = entry['name']
    apo = entry.get('apo', False)
    return name + '_apo' if apo else name


# ── Status checkers ────────────────────────────────────────────────────────
def check_build(sysname):
    """Check if the build stage is complete, running, or failed.
    
    Detects orphaned simrunning markers by checking for evidence that the
    build job already finished (LSF logs, jobqueues.done marker, or stale
    simrunning file with no active LSF job).
    """
    build_dir = os.path.join(RESULTSPATH, 'build', sysname)
    structure_pdb = os.path.join(build_dir, 'structure.pdb')
    structure_psf = os.path.join(build_dir, 'structure.psf')
    simrunning = os.path.join(build_dir, 'simrunning')
    
    if os.path.exists(structure_pdb) and os.path.exists(structure_psf):
        return STATUS_DONE
    
    if os.path.exists(simrunning):
        # Check if the build job actually finished (orphaned simrunning)
        jobqueues_done = os.path.join(build_dir, 'jobqueues.done')
        
        # Evidence the LSF job completed (jobqueues.done exists from trap)
        if os.path.exists(jobqueues_done):
            # jobqueues.done exists but no build products → LSF job finished, build failed
            return STATUS_FAILED
        
        # Check LSF error logs for evidence of failure
        lsf_errs = list(Path(build_dir).glob('lsf.*.err'))
        if lsf_errs:
            for errf in lsf_errs:
                if errf.stat().st_size > 0:
                    return STATUS_FAILED
        
        # Check if simrunning is stale (>24h old) with no active build job
        try:
            mtime = os.path.getmtime(simrunning)
            age_hours = (time.time() - mtime) / 3600
            if age_hours > 24:
                # Check if there's an active LSF job for this build
                # (Look for job.sh which indicates LSF submission)
                job_sh = os.path.join(build_dir, 'job.sh')
                if os.path.exists(job_sh):
                    # LSF job was submitted — check if it's still queued
                    jobname = f'{sysname}_bld'[:30]
                    if not _job_already_queued(jobname):
                        # LSF job completed/failed but simrunning orphaned
                        return STATUS_FAILED
                else:
                    # No LSF job, no job.sh — build ran locally and must have
                    # crashed without cleaning up simrunning
                    return STATUS_FAILED
        except Exception:
            pass
        
        return STATUS_RUNNING
    
    if os.path.exists(build_dir) and os.path.exists(os.path.join(build_dir, 'run.sh')):
        # Job was submitted but may have failed — check LSF logs
        lsf_errs = list(Path(build_dir).glob('lsf.*.err'))
        if lsf_errs:
            for errf in lsf_errs:
                if errf.stat().st_size > 0:
                    return STATUS_FAILED
        # If there's a job.sh (LSF wrapper), the job was submitted via LSF.
        # If we have LSF logs but no error content, check if job is still queued.
        job_sh = os.path.join(build_dir, 'job.sh')
        if os.path.exists(job_sh):
            lsf_outs = list(Path(build_dir).glob('lsf.*.out'))
            if lsf_outs:
                # LSF job ran and produced output → check if it failed
                jobname = f'{sysname}_bld'[:30]
                if not _job_already_queued(jobname):
                    # Job finished but no error in logs — may have succeeded partially
                    # but without build products, treat as failed
                    return STATUS_FAILED
        # No LSF logs at all and no simrunning → setup done but never executed
        if not lsf_errs and not os.path.exists(os.path.join(build_dir, 'simrunning')):
            return STATUS_PENDING
        return STATUS_RUNNING  # queued or running
    # Empty directory or only leftover files from a failed submission → retry
    return STATUS_PENDING


def check_equil(sysname):
    """Check if equilibration is complete, partial, running, or failed."""
    equil_dir = os.path.join(RESULTSPATH, 'equil', sysname)
    simrunning = os.path.join(equil_dir, 'simrunning')
    output_xtc = os.path.join(equil_dir, 'output.xtc')
    log_file = os.path.join(equil_dir, 'log.txt')
    
    # Completed: output.xtc exists AND log confirms successful completion
    if os.path.exists(output_xtc) and is_simulation_done(log_file):
        return STATUS_DONE
    # Running: simrunning flag exists (job is currently executing)
    if os.path.exists(simrunning):
        return STATUS_RUNNING
    # Running: job is in LSF queue (RUN/PEND) even if simrunning not yet visible
    if _job_already_queued(f'{sysname}_eql'[:30]):
        return STATUS_RUNNING
    # Partial: output.xtc exists but log doesn't show completion (job was cut off)
    if os.path.exists(output_xtc):
        return STATUS_PARTIAL
    if Path(equil_dir, 'setup.ready').exists():
        return STATUS_PENDING
    # Submitted but may have failed — check LSF logs
    if os.path.exists(equil_dir) and os.path.exists(os.path.join(equil_dir, 'run.sh')):
        lsf_errs = list(Path(equil_dir).glob('lsf.*.err'))
        if lsf_errs:
            for errf in lsf_errs:
                if errf.stat().st_size > 0:
                    return STATUS_FAILED
        # If there's a job.sh (LSF wrapper), the job was submitted via LSF.
        # If we have LSF output logs, the job actually ran — check if still queued.
        job_sh = os.path.join(equil_dir, 'job.sh')
        if os.path.exists(job_sh):
            lsf_outs = list(Path(equil_dir).glob('lsf.*.out'))
            if lsf_outs:
                # LSF job ran and produced output → check if it's still queued
                jobname = f'{sysname}_eql'[:30]
                if not _job_already_queued(jobname):
                    # Job finished but no products — treat as failed
                    return STATUS_FAILED
        # No LSF logs at all and no simrunning → setup done but never submitted
        if not lsf_errs and not os.path.exists(simrunning):
            return STATUS_PENDING
        return STATUS_RUNNING  # queued or running
    return STATUS_PENDING


def check_production(sysname, rep=None):
    """Check production status. If rep is None, returns dict of all reps."""
    prod_dir = os.path.join(RESULTSPATH, 'production', sysname)
    
    if rep is not None:
        rep_dir = os.path.join(prod_dir, f'rep_{rep}')
        simrunning = os.path.join(rep_dir, 'simrunning')
        output_xtc = os.path.join(rep_dir, 'output.xtc')
        log_file = os.path.join(rep_dir, 'log.txt')
        
        # Completed: output.xtc exists AND log confirms successful completion
        if os.path.exists(output_xtc) and is_simulation_done(log_file):
            return STATUS_DONE
        # Running: simrunning flag exists
        if os.path.exists(simrunning):
            return STATUS_RUNNING
        # Running: job is in LSF queue (RUN/PEND) even if simrunning not yet visible
        if _job_already_queued(f'{sysname}_pr{rep}'[:30]):
            return STATUS_RUNNING
        # Partial: output.xtc exists but log doesn't show completion (job was cut off)
        if os.path.exists(output_xtc):
            return STATUS_PARTIAL
        if Path(rep_dir, 'setup.ready').exists():
            return STATUS_PENDING
        # Submitted but may have failed — check LSF logs
        if os.path.exists(rep_dir) and os.path.exists(os.path.join(rep_dir, 'run.sh')):
            lsf_errs = list(Path(rep_dir).glob('lsf.*.err'))
            if lsf_errs:
                for errf in lsf_errs:
                    if errf.stat().st_size > 0:
                        return STATUS_FAILED
            # If there's a job.sh (LSF wrapper), the job was submitted via LSF.
            # If we have LSF output logs, the job actually ran — check if still queued.
            job_sh = os.path.join(rep_dir, 'job.sh')
            if os.path.exists(job_sh):
                lsf_outs = list(Path(rep_dir).glob('lsf.*.out'))
                if lsf_outs:
                    # LSF job ran and produced output → check if it's still queued
                    jobname = f'{sysname}_pr{rep}'[:30]
                    if not _job_already_queued(jobname):
                        # Job finished but no products — treat as failed
                        return STATUS_FAILED
            # No LSF logs at all and no simrunning → setup done but never submitted
            if not lsf_errs and not os.path.exists(simrunning):
                return STATUS_PENDING
            return STATUS_RUNNING
        return STATUS_PENDING
    
    # Return dict of all replicates
    return {r: check_production(sysname, r) for r in range(1, REPNUM + 1)}


def check_ionize(sysname):
    """Check if ionization is complete (intermediate build step)."""
    ionize_dir = os.path.join(RESULTSPATH, 'ionize', sysname)
    structure_psf = os.path.join(ionize_dir, 'structure.psf')
    if os.path.exists(structure_psf):
        return STATUS_DONE
    return STATUS_PENDING


# ── Status summary ─────────────────────────────────────────────────────────
def get_all_status(targets):
    """Get status of all targets across all stages."""
    status = {}
    for entry in targets:
        sysname = get_sysname(entry)
        build = check_build(sysname)
        equil = check_equil(sysname) if build == STATUS_DONE else STATUS_PENDING
        prod = check_production(sysname) if equil == STATUS_DONE else {r: STATUS_PENDING for r in range(1, REPNUM + 1)}
        
        status[sysname] = {
            'name': entry['name'],
            'pdbcode': entry['pdbcode'],
            'build': build,
            'equil': equil,
            'prod': prod,
        }
    return status


def print_status(status):
    """Pretty-print the pipeline status."""
    # Count summary
    counts = defaultdict(int)
    for s in status.values():
        counts['total'] += 1
        counts['build_' + s['build']] += 1
        counts['equil_' + s['equil']] += 1
        if isinstance(s['prod'], dict):
            for r, st in s['prod'].items():
                counts[f'prod_rep{r}_{st}'] += 1
    
    print(f"\n{'='*90}")
    print(f"  PIPELINE STATUS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*90}")
    # GPU queue status
    gpu_counts = count_gpu_jobs()
    
    print(f"  Total targets: {counts['total']}")
    print(f"  Build:   {counts.get('build_done', 0)} done, {counts.get('build_running', 0)} running, "
          f"{counts.get('build_failed', 0)} failed, {counts.get('build_pending', 0)} pending")
    print(f"  Equil:   {counts.get('equil_done', 0)} done, {counts.get('equil_running', 0)} running, "
          f"{counts.get('equil_failed', 0)} failed, {counts.get('equil_partial', 0)} partial, "
          f"{counts.get('equil_pending', 0)} pending")
    for r in range(1, REPNUM + 1):
        print(f"  Prod r{r}:  {counts.get(f'prod_rep{r}_done', 0)} done, "
              f"{counts.get(f'prod_rep{r}_running', 0)} running, "
              f"{counts.get(f'prod_rep{r}_failed', 0)} failed, "
              f"{counts.get(f'prod_rep{r}_partial', 0)} partial, "
              f"{counts.get(f'prod_rep{r}_pending', 0)} pending")
    if gpu_counts:
        print(f"  GPU queue: {gpu_counts['total']}/{MAX_GPU_JOBS} max — "
              f"{gpu_counts['running']} RUN, {gpu_counts['pending']} PEND")
    print(f"{'='*90}")
    
    # Per-target detail
    print(f"\n{'System':<40s} {'PDB':<6s} {'Build':<10s} {'Equil':<10s} {'Rep1':<10s} {'Rep2':<10s} {'Rep3':<10s}")
    print('-' * 96)
    
    for sysname, s in sorted(status.items()):
        prod_strs = []
        if isinstance(s['prod'], dict):
            for r in range(1, REPNUM + 1):
                st = s['prod'].get(r, STATUS_PENDING)
                icon = {'done': '✅', 'running': '🔄', 'failed': '❌', 'pending': '⏳', 'partial': '🔶'}.get(st, '❓')
                prod_strs.append(f'{icon} {st:<7s}')
        else:
            prod_strs = ['⏳ pending'] * REPNUM
        
        build_icon = {'done': '✅', 'running': '🔄', 'failed': '❌', 'pending': '⏳', 'partial': '🔶'}.get(s['build'], '❓')
        equil_icon = {'done': '✅', 'running': '🔄', 'failed': '❌', 'pending': '⏳', 'partial': '🔶'}.get(s['equil'], '❓')
        
        print(f"{sysname:<40s} {s['pdbcode']:<6s} "
              f"{build_icon} {s['build']:<7s}  "
              f"{equil_icon} {s['equil']:<7s}  "
              f"{prod_strs[0]:<12s} {prod_strs[1]:<12s} {prod_strs[2]:<12s}")
    print()


# ── GPU queue monitoring ──────────────────────────────────────────────────
def count_gpu_jobs():
    """Count currently running + pending GPU jobs via bjobs.
    Returns dict with 'running', 'pending', 'total' counts, or None if bjobs fails."""
    try:
        result = subprocess.run(
            ['bjobs', '-q', LSF_QUEUE, '-o', 'stat', '-noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None
        stats = result.stdout.strip().split('\n')
        running = sum(1 for s in stats if s.strip() == 'RUN')
        pending = sum(1 for s in stats if s.strip() in ('PEND', 'WAIT'))
        return {'running': running, 'pending': pending, 'total': running + pending}
    except Exception as e:
        print(f"  WARNING: Could not query GPU queue: {e}")
        return None


def wait_for_gpu_slot(max_jobs=MAX_GPU_JOBS, poll_interval=30, timeout=None):
    """Block until GPU queue has fewer than max_jobs running+pending.
    Returns True if slot is available, False on timeout."""
    start = time.time()
    while True:
        counts = count_gpu_jobs()
        if counts is None:
            print("  WARNING: Cannot query bjobs, proceeding anyway...")
            return True
        if counts['total'] < max_jobs:
            return True
        elapsed = time.time() - start
        print(f"  ⏳ GPU queue full ({counts['total']}/{max_jobs}): "
              f"{counts['running']} RUN, {counts['pending']} PEND. "
              f"Waiting {poll_interval}s... (elapsed: {int(elapsed)}s)")
        if timeout is not None and elapsed > timeout:
            print(f"  ⚠️  Timeout waiting for GPU slot after {int(elapsed)}s")
            return False
        time.sleep(poll_interval)


# ── Duplicate job detection ────────────────────────────────────────────────
def _job_already_queued(jobname):
    """Check if a job with the given name is already in the LSF queue (RUN or PEND).
    Returns True if a duplicate exists, False otherwise."""
    try:
        result = subprocess.run(
            ['bjobs', '-J', jobname, '-o', 'stat', '-noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return False  # bjobs failed — assume not duplicate
        stats = result.stdout.strip().split('\n')
        for s in stats:
            if s.strip() in ('RUN', 'PEND', 'WAIT'):
                return True
        return False
    except Exception:
        return False  # can't check — proceed cautiously


def _check_and_prevent_duplicate(stage_dir, jobname):
    """Check for duplicate job by LSF queue and simrunning marker.
    Returns (is_duplicate: bool, reason: str)."""
    # Check simrunning marker (created by job_commands prerun)
    simrunning = os.path.join(stage_dir, 'simrunning')
    if os.path.exists(simrunning):
        return True, f"simrunning marker exists in {stage_dir}"
    # Check LSF queue for existing job with same name
    if _job_already_queued(jobname):
        return True, f"job '{jobname}' already in LSF queue (RUN/PEND)"
    return False, ""


# ── LSF job submission ─────────────────────────────────────────────────────
def submit_lsf_job(stage_dir, jobname, walltime, prerun_commands=None):
    """
    Submit a job via LSF. Returns True if submission succeeded.
    Uses the jobqueues.lsfqueue module from the GPCRmd conda environment.
    """
    try:
        # We run submission via the conda python to ensure proper environment

        run_command = '\n'.join([
            'trap - EXIT SIGTERM',
            'rm -f {0}/jobqueues.done {0}/jobqueues.failed'.format(stage_dir),
            'touch {0}/simrunning'.format(stage_dir),
            "trap 'rm -f {0}/simrunning; touch {0}/jobqueues.failed; exit 143' TERM INT".format(stage_dir),
            'cd {0}'.format(stage_dir),
            'bash run.sh',
            'status=$?',
            'rm -f {0}/simrunning'.format(stage_dir),
            "if [ \"$status\" -eq 0 ] && grep -q '# Simulation completed!' log.txt; then",
            '    touch {0}/jobqueues.done'.format(stage_dir),
            'else',
            '    touch {0}/jobqueues.failed'.format(stage_dir),
            '    if [ \"$status\" -eq 0 ]; then status=1; fi',
            'fi',
            'exit \"$status\"',
        ])
        submit_script = f'''
import os, sys
sys.path.insert(0, '{BASEPATH}')
from jobqueues.lsfqueue import LsfQueue

sq = LsfQueue()
sq.version = 10
sq.walltime = {walltime}
sq.resources = ["rusage[mem={LSF_MEM}] span[hosts=1] select[ncc>=8.0]"]
sq.envvars = "{ACEMD_LICENSE}"
sq.jobname = "{jobname}"
sq.datadir = None
sq.queue = "{LSF_QUEUE}"
sq.ngpu = {LSF_NGPU}
sq.gpu_options = {{'mode': 'exclusive_process'}}
sq.ncpu = {LSF_NCPU}
'''
        if prerun_commands:
            submit_script += f'sq.prerun = {prerun_commands}\n'
        
        submit_script += f'sq.submit({stage_dir!r}, commands=[{run_command!r}])\n'
        submit_script += "print('SUBMITTED')\n"
        
        result = subprocess.run(
            [CONDA_PYTHON, '-c', submit_script],
            capture_output=True, text=True, timeout=30,
            cwd=BASEPATH
        )
        
        if 'SUBMITTED' in result.stdout:
            return True
        else:
            print(f"  LSF submission warning: {result.stderr.strip()}")
            return 'submitted' in result.stdout.lower() or result.returncode == 0
    except Exception as e:
        print(f"  LSF submission error: {e}")
        return False


def _prefetch_network_data(sysname, entry, build_dir):
    """Pre-fetch network-dependent data on the controller node and cache to disk.
    
    This ensures compute nodes (which may lack internet access) can build
    systems using only local files.
    
    Caches:
      - GPCRdb structure list → RESULTSPATH/gpcrdb_cache.json (once)
      - OPM thickness + structure → build_dir/opm_thickness.json + build_dir/opm_structure.pdb
    """
    import requests
    
    pdbcode = entry['pdbcode']
    gpcrdb_cache = os.path.join(RESULTSPATH, 'gpcrdb_cache.json')
    opm_thickness_file = os.path.join(build_dir, 'opm_thickness.json')
    opm_pdb_file = os.path.join(build_dir, 'opm_structure.pdb')
    
    # ── GPCRdb cache (once for all targets) ──
    if not os.path.exists(gpcrdb_cache):
        print(f"  📡 Fetching GPCRdb structure data...")
        try:
            gpcrdb_data = requests.get('http://gpcrdb.org/services/structure/', timeout=30).json()
            gpcrdb_dict = {entry2['pdb_code']: entry2 for entry2 in gpcrdb_data}
            with open(gpcrdb_cache, 'w') as f:
                json.dump(gpcrdb_dict, f)
            print(f"  ✅ GPCRdb cache saved ({len(gpcrdb_dict)} entries)")
        except Exception as e:
            print(f"  ⚠️  GPCRdb fetch failed: {e} — will use empty cache")
            with open(gpcrdb_cache, 'w') as f:
                json.dump({}, f)
    
    # ── OPM data per target ──
    if not os.path.exists(opm_thickness_file) or not os.path.exists(opm_pdb_file):
        print(f"  📡 Fetching OPM data for {pdbcode}...")
        try:
            headers = {
                'Accept': 'application/json, text/plain, */*',
                'Accept-Encoding': 'gzip, deflate, br, zstd',
                'Origin': 'https://opm.phar.umich.edu',
                'Referer': 'https://opm.phar.umich.edu/'
            }
            # Query OPM for this PDB code
            searchlink = ('https://opm-back.cc.lehigh.edu/opm-backend//primary_structures'
                          f'?search={pdbcode}&sort=&pageSize=100')
            response_dict = requests.get(searchlink, headers=headers, timeout=30).json()
            
            # Fallback to 4EJ4 if no OPM entry for this PDB
            if not len(response_dict['objects']):
                searchlink = ('https://opm-back.cc.lehigh.edu/opm-backend//primary_structures'
                              '?search=4EJ4&sort=&pageSize=100')
                response_dict = requests.get(searchlink, headers=headers, timeout=30).json()
            
            thickness = response_dict['objects'][0]['thickness']
            opm_id = response_dict['objects'][0]['id']
            
            # Get the actual PDB ID used by OPM
            opmlink = f'https://opm-back.cc.lehigh.edu/opm-backend//primary_structures/{opm_id}'
            response_dict = requests.get(opmlink, headers=headers, timeout=30).json()
            new_pdbcode = response_dict['pdbid'].lower()
            
            # Download OPM PDB structure (filter out DUM residues)
            response_pdb = requests.get(
                f'https://opm-assets.storage.googleapis.com/pdb/{new_pdbcode}.pdb',
                headers=headers, timeout=30
            )
            line_list = response_pdb.content.decode('UTF-8').split('\n')
            with open(opm_pdb_file, 'w') as f:
                for line in line_list:
                    if ' DUM ' not in line:
                        f.write(line + '\n')
            
            # Save thickness
            with open(opm_thickness_file, 'w') as f:
                json.dump({'thickness': thickness}, f)
            
            print(f"  ✅ OPM data cached for {pdbcode} (thickness={thickness})")
        except Exception as e:
            print(f"  ⚠️  OPM fetch failed for {pdbcode}: {e}")
            # Don't create empty files — let the build script handle the error properly


def setup_build(sysname, entry):
    """Prepare and run a build job for a single target locally.
    
    The build runs on the local machine and handles:
    - Internal waters (GPCR-specific)
    - Ligand replacement with parametrized version
    - Fix & prepare input (segids, protonation)
    - OPM alignment via Chimera
    - Protein preparation (protonation states, sidechain optimization)
    - Membrane placement (POPC)
    - Solvation (TIP3 water)
    - CHARMM topology assignment & ionization
    """
    build_dir = os.path.join(RESULTSPATH, 'build', sysname)
    
    # Check if already built
    if os.path.exists(os.path.join(build_dir, 'structure.pdb')) and \
       os.path.exists(os.path.join(build_dir, 'structure.psf')):
        print(f"  Build already exists for {sysname}, skipping.")
        return True
    
    os.makedirs(build_dir, exist_ok=True)
    
    # Write a simrunning marker so status shows RUNNING
    with open(os.path.join(build_dir, 'simrunning'), 'w') as f:
        f.write(f"Build submitted at {datetime.now().isoformat()}\n")
    
    # ── Pre-fetch network data on controller node before submitting ──
    try:
        _prefetch_network_data(sysname, entry, build_dir)
    except Exception as e:
        print(f"  ⚠️  Network prefetch failed for {sysname}: {e}")
        print(f"  ⚠️  Will proceed without cached data — compute node may need network fallbacks")
    
    # Build the run.sh script — this is a self-contained Python script
    # that replicates the Part 2 build logic from simulate_gpcr_agonist_all.ipynb
    # Build runs locally so network access is available.
    run_script = f'''#!/bin/bash
cd {BASEPATH}
export PATH={BASEPATH}notebooks:$PATH
{CONDA_PYTHON} << 'PYEOF'
import os, sys, json, time, traceback
os.chdir("{BASEPATH}")
sys.path.insert(0, "{BASEPATH}notebooks")

from simulate_structures_functions import *
from moleculekit.writers import PDBwrite
from htmd.builder.charmm import _recoverProtonations
import charmm
from pathlib import Path

# --- Load entry ---
with open("{INPUT_JSON}") as f:
    all_entries = json.load(f)
entry = None
for e in all_entries:
    if e["name"] == "{entry['name']}":
        entry = e
        break
if entry is None:
    raise ValueError("Entry {entry['name']} not found in JSON")

name = entry["name"]
isgpcr = entry["isgpcr"]
lig_info = entry["ligands"]
pdbcode = entry["pdbcode"]
pdbfile = entry["pdbfile"]
curated = entry["curated"]
sod2x50 = entry["sod2x50"]
prot_chain = entry["prot_chain"]
gpcr_chain = entry["prot_chain"] if isgpcr else False
apo = entry["apo"]

start_time = time.time()
sysname = name + "_apo" if apo else name

# --- Load cached network data (pre-fetched on controller) ---
build_dir = "{build_dir}"
resultspath = "{RESULTSPATH}"
BASEPATH = "{BASEPATH}"
# --- Load topology files ---
topparpath = "{BASEPATH}toppar/TOP_PARAMS_ACE3/"
topos = [os.path.join(topparpath, str(f)) for f in Path(topparpath).rglob("*.rtf")]
params = [os.path.join(topparpath, str(f)) for f in Path(topparpath).rglob("*.prm")]
streams = [os.path.join(topparpath, str(f)) for f in Path(topparpath).rglob("*.str")]

# --- Load parametrization data ---
modresdict_path = "{BASEPATH}modified_residues.json"
ligandsdict_path = "{BASEPATH}ligands.json"
modresdict = json_dict(modresdict_path) if os.path.exists(modresdict_path) else {{}}
ligandsdict = json_dict(ligandsdict_path) if os.path.exists(ligandsdict_path) else {{}}

# --- Paths ---
membranepdb = "{BASEPATH}membrane/popc36_box_renumbered.pdb"
psfgenpath = "/home/ji.cheng4-umw/NAMD_3.0.2/psfgen"

# --- Build the system ---
try:
    # Aminergic / adenosine check (gpcrdb_dict loaded from cache)
    if isgpcr and gpcrdb_dict:
        try:
            aminergic = gpcrdb_dict.get(pdbcode, {{}}).get("family", "").startswith("001_001")
            adenosine = gpcrdb_dict.get(pdbcode, {{}}).get("family", "").startswith("001_006_001")
        except Exception:
            aminergic = False; adenosine = False
    elif isgpcr:
        print("WARNING: gpcrdb_dict unavailable (network issue), using defaults for {{name}}")
        aminergic = False; adenosine = False
    else:
        aminergic = None; adenosine = None

    # Internal waters (GPCR-specific)
    if isgpcr:
        (sod2x50, watered_filename) = internal_waters(pdbfile, pdbcode, apo, sod=sod2x50)
    else:
        watered_filename = pdbfile
    if not os.path.exists(watered_filename):
      # HomolWat fallback: if no _HW.pdb was produced, use the original PDB
      # returned by internal_waters() instead of failing on a missing file.
      # (fallback assignment is the body below)
          watered_filename = pdbfile

    mol = Molecule(watered_filename)
    print(f"Loaded: {{watered_filename}}")

    # Remove crystallization artifacts
    mol.remove("resname " + " ".join(blacklist))

    # Replace every declared ligand independently at its source position.
    # Removing all of chain L and appending only lig_info[0] silently dropped
    # co-ligands in multi-ligand targets such as NTSR1_8ZYT.
    if not apo and lig_info:
        print("Replacing declared ligands...")
        for lig_number, lig_entry in enumerate(lig_info, start=1):
            lig_resname = lig_entry["resname"]
            lig_sel = 'resname "' + lig_resname + '"'
            lig_indices = mol.get("index", sel=lig_sel)
            if len(lig_indices) == 0:
                raise ValueError("Declared ligand " + lig_resname + " is absent from " + watered_filename)
            lig_center = mol.getCenter(lig_sel)
            lig_mol_dir = "{BASEPATH}toppar/Ligands/" + lig_resname + "/" + name + "_" + lig_resname + ".mol2"
            if not os.path.exists(lig_mol_dir):
                raise FileNotFoundError("Missing ligand MOL2: " + lig_mol_dir)
            lig_mol = Molecule(lig_mol_dir)
            lig_mol.set("resid", lig_number, sel="all")
            atom_name_map = lig_entry.get("atom_name_map")
            if atom_name_map:
                from ligand_geometry import align_named_coordinates
                fitted, pose_rmsd = align_named_coordinates(
                    lig_mol.coords[:, :, 0], lig_mol.name,
                    mol.coords[lig_indices, :, 0], mol.name[lig_indices],
                    atom_name_map,
                )
                if pose_rmsd > 0.5:
                    raise ValueError("Ligand differs from mapped bound pose by more than 0.5 A")
                lig_mol.coords[:, :, 0] = fitted
                print("Mapped ligand bound-pose RMSD:", pose_rmsd)
            else:
                lig_mol.center(lig_center)
            mol.remove(lig_sel)
            mol.append(lig_mol, collisions=False, coldist=1.3, removesel="all", invertcollisions=False)
            mol.set("resname", lig_resname, "resname UNL")
            mol.set("segid", "LIG", sel=lig_sel)
            mol.set("chain", "L", sel=lig_sel)

    # Remove sodium from non-A-class GPCRs
    if isgpcr:
        try:
            is_class_a = gpcrdb_dict.get(pdbcode, {{}}).get("family", "").startswith("001")
        except Exception:
            is_class_a = True  # assume class A (keep sodium) if data unavailable
        if not is_class_a:
            mol.remove("element NA")

    print(f"{{name}} {{pdbcode}}")
    mol_fixed, prot_segids = fix_and_prepare_input(mol, name, pdbcode, modresdict,
                                                    isgpcr=isgpcr, prot_chain=prot_chain)

    # Handle covalent ligands or apo
    if apo or not lig_info:
        (mod_mol, prot_segids) = make_apo(mol_fixed, prot_chain)
        covligs = []
    else:
        (mod_mol, covligs) = covalent_ligands(mol_fixed, name, ligandsdict)

    # Get aligned OPM structure (network access available since build runs locally)
    thickness, opm_mol = get_opm(pdbcode)

    # Superimpose onto OPM via Chimera
    mod_mol.write("mod_mol.pdb")
    opm_mol.write("opm_mol.pdb")
    chimera_superimpose("opm_mol.pdb", "mod_mol.pdb", "aligned_mol.pdb")
    mol_aligned = Molecule("aligned_mol.pdb")

    # Center to receptor XY
    center = np.mean(mol_aligned.get("coords", sel="chain " + prot_chain), axis=0)
    mol_aligned.moveBy([-center[0], -center[1], 0])

    # Prepare protein
    if curated:
        prepared_mol = mol_aligned
    else:
        prepared_mol = prepare_system(mol_aligned, pdbcode, thickness, gpcr_chain, sod2x50, aminergic, adenosine)
    prepared_mol.write("prepared_mol.pdb")

    # Add membrane
    print("Adding membrane...")
    membranemol = Molecule(membranepdb)
    mol_membraned, membrane_resnames, membrane_segids, xreps, yreps = add_membrane(
        prepared_mol, membranemol, prot_segids, membrane_distance)

    # Solvate
    print("Solvating...")
    mol_solvated = solvate_pdbmol(mol_membraned, membrane_segids,
                                   water_thickness, water_margin,
                                   buffer=buffer, coldist=coldist, prefix="WT")
    mol_solvated.write("mol_solvated.pdb")

    # Extra parameters
    ligstreams = extra_parameters(name, ligandsdict, modresdict, blacklist, covligs, basepath="{BASEPATH}")
    print(ligstreams)

    # Cap atoms
    caps = get_caps(prot_segids, mol_solvated)

    # Rename ligand atoms
    if not apo and lig_info:
        mol_solvated = rename_ligand_atoms_from_mol2(mol_solvated, name, ligandsdict, "{BASEPATH}")

    # Pre-build
    print("Pre-build...")
    prebuildmol = charmm.build(mol_solvated,
                                topo=topos + streams + ligstreams,
                                caps=caps,
                                outdir=resultspath + "pre-build/" + sysname,
                                ionize=False,
                                psfgen=psfgenpath)

    prebuild_pdbfile = os.path.splitext(prebuildmol.topoloc)[0] + ".pdb"
    prebuildmol = Molecule(prebuild_pdbfile, validateElements=False)
    _recoverProtonations(prebuildmol)

    # Check water/lipid ratio
    lipid_num = len(set(prebuildmol.get("resid", sel="segid " + membrane_lipid_segid)))
    solv_num = len(prebuildmol.get("index", sel="resname TIP3 and name OH2"))
    if float(solv_num) / lipid_num < 35:
        raise ValueError("Water/lipid ratio lower than 35.")

    # Renumber residues
    print("Renumbering...")
    mol_renumbered = renumber_resid_vmd(prebuildmol, "segid " + " ".join(membrane_segids), by=2)

    # Ionize
    print("Ionizing...")
    molbuilt = charmm.build(prebuildmol,
                             topo=topos + streams + ligstreams,
                             outdir=resultspath + "ionize/" + sysname,
                             saltconc=0.15,
                             caps=caps,
                             psfgen=psfgenpath)

    build_psffile = molbuilt.topoloc
    build_pdbfile = os.path.splitext(molbuilt.topoloc)[0] + ".pdb"
    molbuilt = Molecule(build_pdbfile, validateElements=False)
    _recoverProtonations(molbuilt)

    # Final build
    print("Building...")
    molbuilt = renumber_resid_vmd(molbuilt, 'segid "WT.*" or segid I', by=2)
    molbuilt = charmm.build(molbuilt,
                             topo=topos + streams + ligstreams,
                             outdir=resultspath + "build/" + sysname,
                             ionize=False,
                             psfgen=psfgenpath)

    build_path = resultspath + "build/" + sysname + "/"
    # Split ligand stream files if any extra parameters exist
    # Every ligand stream needs its own parameter file (some targets have >1).
    for ligstream in ligstreams:
        charmm.split(ligstream, build_path)
    # Combine all main parameter files into parameters.prm
    charmm.combine(params, build_path + "parameters.prm")
    # Find any additional .prm files generated by split (exclude parameters.prm itself)
    tmp_prmfilenames = [f for f in Path(build_path).rglob("*.prm")
                        if f.name != "parameters.prm"]
    if tmp_prmfilenames:
        tmp_params = [os.path.join(build_path, str(f)) for f in tmp_prmfilenames]
        charmm.combine(tmp_params + [build_path + "parameters.prm"], build_path + "parameters.prm")

    print(f"End of {{sysname}} after {{time.time() - start_time}} seconds")

    # Write done marker
    with open("{build_dir}/jobqueues.done", "w") as f:
        f.write("Build completed successfully\\n")

except Exception as e:
    print(f"model {{sysname}} could not be built because {{e}}")
    print(traceback.format_exc())
    sys.exit(1)
PYEOF
'''
    with open(os.path.join(build_dir, 'run.sh'), 'w') as f:
        f.write(run_script)
    
    os.chmod(os.path.join(build_dir, 'run.sh'), 0o755)
    
    # Run build locally (no LSF submission) and keep a persistent live log.
    print(f"  🔨 Running build locally: {sysname}")
    log_file = os.path.join(build_dir, 'log.txt')
    try:
        with open(log_file, 'w') as build_log:
            result = subprocess.run(
                ['bash', os.path.join(build_dir, 'run.sh')],
                stdout=build_log, stderr=subprocess.STDOUT, text=True,
                cwd=build_dir
            )
        simrunning_file = os.path.join(build_dir, 'simrunning')
        if result.returncode == 0:
            # Remove simrunning marker on success
            if os.path.exists(simrunning_file):
                os.remove(simrunning_file)
            print(f"  ✅ Build completed: {sysname}")
            return True
        else:
            # Remove simrunning marker on failure so status shows correctly
            if os.path.exists(simrunning_file):
                os.remove(simrunning_file)
            with open(log_file, 'r', errors='replace') as build_log:
                build_log.seek(0, 2)
                build_log.seek(max(0, build_log.tell() - 4000))
                failure_tail = build_log.read()
            print(f"  ❌ Build failed for {sysname} (exit code {result.returncode})")
            print(f"  Log tail:\n{failure_tail}")
            return False
    except Exception as e:
        # Remove simrunning marker on exception so status shows correctly
        simrunning_file = os.path.join(build_dir, 'simrunning')
        if os.path.exists(simrunning_file):
            os.remove(simrunning_file)
        print(f"  ❌ Build error for {sysname}: {e}")
        traceback.print_exc()
        return False


def setup_equilibration(sysname, submit=True):
    """Prepare and submit equilibration for a target.
    
    Detects partial runs (output.xtc exists but log doesn't show completion)
    and resumes from checkpoint instead of regenerating input files.
    """
    equil_dir = os.path.join(RESULTSPATH, 'equil', sysname)
    build_dir = os.path.join(RESULTSPATH, 'build', sysname)
    
    if not os.path.exists(os.path.join(build_dir, 'structure.pdb')):
        print(f"  ERROR: Build not complete for {sysname}")
        return False
    
    # ── Duplicate prevention ──
    jobname = f'{sysname}_eql'[:30]
    is_dup, reason = _check_and_prevent_duplicate(equil_dir, jobname)
    if is_dup:
        print(f"  ⚠️  Skipping {sysname} equil: {reason}")
        return False
    
    os.makedirs(equil_dir, exist_ok=True)

    if not _ensure_charmm_parameters(build_dir):
        return False
    
    # Check if this is a partial resume (output.xtc exists but simulation not complete)
    output_xtc = os.path.join(equil_dir, 'output.xtc')
    log_file = os.path.join(equil_dir, 'log.txt')
    is_partial = os.path.exists(output_xtc) and not is_simulation_done(log_file)
    is_already_done = os.path.exists(output_xtc) and is_simulation_done(log_file)

    recovered_from_nan = (not os.path.exists(os.path.join(equil_dir, 'input.yaml'))
                          and any(Path(equil_dir).glob('failed_nan_*')))
    # A first-step NaN happens before output.xtc is created, so failed jobs as
    # well as partial jobs must be inspected for this recovery condition.
    if not is_already_done and os.path.exists(log_file):
        with open(log_file, 'r', errors='replace') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 20000))
            recovered_from_nan = recovered_from_nan or 'Particle coordinate is NaN' in f.read()

    if recovered_from_nan:
        failed_dir = os.path.join(
            equil_dir, 'failed_nan_' + datetime.now().strftime('%Y%m%d-%H%M%S')
        )
        os.makedirs(failed_dir)
        for pattern in (
            'input.yaml', 'run.sh', 'job.sh', 'jobqueues.failed', 'log.txt',
            'minimized.coor', 'output.csv', 'output.xtc', 'restart.chk',
            'restraint_*.sel',
        ):
            for old_path in Path(equil_dir).glob(pattern):
                os.replace(old_path, os.path.join(failed_dir, old_path.name))
        print(f"  Archived unstable checkpoint in {failed_dir}")
        is_partial = False
    
    if is_already_done:
        print(f"  ✅ Equilibration already complete for {sysname}, skipping.")
        return True
    
    if not is_partial and not (Path(equil_dir, 'setup.ready').exists() and Path(equil_dir, 'input.yaml').exists()):
        # ── Fresh setup: generate input.yaml via define_equilibration ──
        setup_script = f'''
import os, sys
os.chdir('{BASEPATH}')
sys.path.insert(0, '{BASEPATH}notebooks')
from acemd.protocols import setup_equilibration

md = setup_equilibration(
    run="40ns", temperature=310, minimize=10000, barostatconstratio=True, timestep=2.0,
    builddir='{build_dir}/',
    outdir='{equil_dir}/'
)

with open('{equil_dir}/run.sh', 'w') as f:
    f.write('#!/bin/bash\\nexec {ACEMD_PATH} > log.txt 2>&1')

print('SETUP_DONE')
'''
        result = subprocess.run(
            [CONDA_PYTHON, '-c', setup_script],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, 'GPCRMD_OFFLINE': '1', 'HTMD_NONINTERACTIVE': '1', 'ACEMD_NONINTERACTIVE': '1'},
            cwd=BASEPATH
        )
        
        if 'SETUP_DONE' not in result.stdout:
            print(f"  Equil setup failed: {result.stderr[:500]}")
            return False

        if recovered_from_nan:
            input_yaml = os.path.join(equil_dir, 'input.yaml')
            with open(input_yaml, 'r') as fy:
                cfg = yaml.safe_load(fy) or {}
            cfg['restart'] = False
            cfg['timestep'] = 1.0
            cfg['minimize'] = max(10000, int(cfg.get('minimize', 0)))
            for restraint in cfg.get('extforces', []):
                setpoints = restraint.get('setpoints', [])
                if setpoints:
                    strength = str(setpoints[-1]).split('@', 1)[0]
                    setpoints[-1] = f'{strength}@20ns'
            with open(input_yaml, 'w') as fy:
                yaml.safe_dump(cfg, fy, sort_keys=False)
            print(f"  Restarting {sysname} fresh at 1 fs after NaN checkpoint failure")
    elif is_partial:
        # ── Partial resume: preserve existing input.yaml, ensure restart:true ──
        print(f"  🔄 Resuming {sysname} from checkpoint (partial run detected)")
        input_yaml = os.path.join(equil_dir, 'input.yaml')
        if os.path.exists(input_yaml):
            with open(input_yaml, 'r') as fy:
                cfg = yaml.safe_load(fy)
            if not cfg.get('restart', False):
                cfg['restart'] = True
                with open(input_yaml, 'w') as fy:
                    yaml.dump(cfg, fy)
                print(f"  📝 Set restart:true in {input_yaml}")
        # Ensure run.sh exists (may have been lost if postrun didn't execute)
        run_sh = os.path.join(equil_dir, 'run.sh')
        if not os.path.exists(run_sh):
            with open(run_sh, 'w') as f:
                f.write(f'#!/bin/bash\nexec {ACEMD_PATH} > log.txt 2>&1')
    
    _configure_charmm_inputs(equil_dir, build_dir)

    # Submit to LSF

    _write_acemd_runner(equil_dir)
    if not submit:
        Path(equil_dir, 'setup.ready').touch()
        return True
    prerun = [f"cd {BASEPATH}", f"mkdir -p {equil_dir}"]
    
    success = submit_lsf_job(equil_dir, jobname, LSF_WALLTIME_EQUIL, prerun)
    if success:
        Path(equil_dir, 'setup.ready').unlink(missing_ok=True)
        print(f"  ✅ Equilibration submitted: {sysname}")
    else:
        print(f"  ❌ Equilibration submission failed: {sysname}")
    return success


def setup_production(sysname, rep, submit=True):
    """Prepare and submit one production replicate.
    
    Detects partial runs and resumes from checkpoint instead of regenerating input files.
    """
    prod_dir = os.path.join(RESULTSPATH, 'production', sysname, f'rep_{rep}')
    equil_dir = os.path.join(RESULTSPATH, 'equil', sysname)
    
    if (not os.path.exists(os.path.join(equil_dir, 'output.xtc'))
            or not is_simulation_done(os.path.join(equil_dir, 'log.txt'))):
        print(f"  ERROR: Equilibration not complete for {sysname}")
        return False
    
    # ── Duplicate prevention ──
    jobname = f'{sysname}_pr{rep}'[:30]
    is_dup, reason = _check_and_prevent_duplicate(prod_dir, jobname)
    if is_dup:
        print(f"  ⚠️  Skipping {sysname} prod rep {rep}: {reason}")
        return False
    
    os.makedirs(prod_dir, exist_ok=True)
    _prepare_production_recovery(prod_dir)
    
    # Check if this is a partial resume (output.xtc exists but simulation not complete)
    output_xtc = os.path.join(prod_dir, 'output.xtc')
    log_file = os.path.join(prod_dir, 'log.txt')
    is_partial = os.path.exists(output_xtc) and not is_simulation_done(log_file)
    is_already_done = os.path.exists(output_xtc) and is_simulation_done(log_file)
    
    if is_already_done:
        print(f"  ✅ Production rep {rep} already complete for {sysname}, skipping.")
        return True
    
    if not is_partial and not (Path(prod_dir, 'setup.ready').exists() and Path(prod_dir, 'input.yaml').exists()):
        # ── Fresh setup: generate input.yaml via define_production ──
        setup_script = f'''
import os, sys
os.chdir('{BASEPATH}')
sys.path.insert(0, '{BASEPATH}notebooks')
from acemd.protocols import setup_production

md = setup_production(
    run="500ns", temperature=310,
    equildir='{equil_dir}/',
    outdir='{prod_dir}/'
)

with open('{prod_dir}/run.sh', 'w') as f:
    f.write('#!/bin/bash\\nexec {ACEMD_PATH} > log.txt 2>&1')

print('SETUP_DONE')
'''
        result = subprocess.run(
            [CONDA_PYTHON, '-c', setup_script],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, 'GPCRMD_OFFLINE': '1', 'HTMD_NONINTERACTIVE': '1', 'ACEMD_NONINTERACTIVE': '1'},
            cwd=BASEPATH
        )
        
        if 'SETUP_DONE' not in result.stdout:
            print(f"  Prod setup failed for {sysname} rep {rep}: {result.stderr[:500]}")
            return False
    elif is_partial:
        # ── Partial resume: preserve existing input.yaml, ensure restart:true ──
        print(f"  🔄 Resuming {sysname} rep {rep} from checkpoint (partial run detected)")
        input_yaml = os.path.join(prod_dir, 'input.yaml')
        if os.path.exists(input_yaml):
            with open(input_yaml, 'r') as fy:
                cfg = yaml.safe_load(fy)
            if not cfg.get('restart', False):
                cfg['restart'] = True
                with open(input_yaml, 'w') as fy:
                    yaml.dump(cfg, fy)
                print(f"  📝 Set restart:true in {input_yaml}")
        # Ensure run.sh exists
        run_sh = os.path.join(prod_dir, 'run.sh')
        if not os.path.exists(run_sh):
            with open(run_sh, 'w') as f:
                f.write(f'#!/bin/bash\nexec {ACEMD_PATH} > log.txt 2>&1')
    
    _configure_charmm_inputs(prod_dir, equil_dir)
    _apply_production_recovery(prod_dir)

    prerun = [f"cd {BASEPATH}", f"mkdir -p {prod_dir}"]

    _write_acemd_runner(prod_dir)
    if not submit:
        Path(prod_dir, 'setup.ready').touch()
        return True
    
    success = submit_lsf_job(prod_dir, jobname, LSF_WALLTIME_PROD, prerun)
    if success:
        Path(prod_dir, 'setup.ready').unlink(missing_ok=True)
        print(f"  ✅ Production rep {rep} submitted: {sysname}")
    else:
        print(f"  ❌ Production rep {rep} submission failed: {sysname}")
    return success


# ── Auto-advance logic ─────────────────────────────────────────────────────
def auto_advance(targets, stage=None, target_name=None, dry_run=False, max_jobs=MAX_GPU_JOBS, wait_for_slot=True):
    """
    Automatically advance targets to the next stage.
    
    Args:
        targets: list of target dicts from JSON
        stage: 'equil' or 'prod' to limit which stage to advance
        target_name: specific target name to advance
        dry_run: if True, only print what would be done
        max_jobs: max GPU jobs allowed before blocking submission
        wait_for_slot: if True, block until GPU slot is available
    """
    status = get_all_status(targets)
    submissions_held = 0

    def recovery_priority(entry):
        current = status[get_sysname(entry)]
        if current['build'] == STATUS_DONE and current['equil'] in (STATUS_PENDING, STATUS_PARTIAL, STATUS_FAILED):
            return 0
        if any(st in (STATUS_PARTIAL, STATUS_FAILED) for st in current['prod'].values()):
            return 1
        if any(Path(RESULTSPATH, 'production', get_sysname(entry), f'rep_{r}', 'recovery.json').exists()
               and current['prod'].get(r) == STATUS_PENDING for r in range(1, REPNUM + 1)):
            return 1
        return 2

    for entry in sorted(targets, key=recovery_priority):
        sysname = get_sysname(entry)
        
        if target_name and entry['name'] != target_name:
            continue
        
        s = status[sysname]
        
        # ── Advance: Pending/Failed → Build ──
        if stage in (None, 'build') and s['build'] in (STATUS_PENDING, STATUS_FAILED):
            current_build = check_build(sysname)
            if current_build not in (STATUS_PENDING, STATUS_FAILED):
                print(f"\n⏭️  Skipping {sysname}: build status changed to '{current_build}' (already started?)")
                continue
            print(f"\n📋 Building {sysname}")
            if dry_run:
                continue
            if not setup_build(sysname, entry):
                continue
            # Refresh status so a successful build can advance to equil in the
            # same pass when no --stage filter was used.
            s = get_all_status([entry])[sysname]

        # ── Advance: Build → Equil (fresh or partial resume) ──
        if (stage in (None, 'equil') and s['build'] == STATUS_DONE
                and s['equil'] in (STATUS_PENDING, STATUS_PARTIAL, STATUS_FAILED)):
            action = {
                STATUS_PARTIAL: "resume equil",
                STATUS_FAILED: "retry equil",
            }.get(s['equil'], "build → equil")
            # ── Re-check status to avoid race-condition duplicates ──
            current_equil = check_equil(sysname)
            if current_equil not in (STATUS_PENDING, STATUS_PARTIAL, STATUS_FAILED):
                print(f"\n⏭️  Skipping {sysname}: equil status changed to '{current_equil}' (already submitted?)")
                continue
            print(f"\n📋 Advancing {sysname}: {action}")
            if not dry_run:
                if _check_gpu_and_wait(wait_for_slot, max_jobs):
                    setup_equilibration(sysname)
                else:
                    submissions_held += 1

        # ── Advance: Equil → Production (fresh or partial resume) ──
        if stage in (None, 'prod') and s['equil'] == STATUS_DONE:
            prod_statuses = s['prod'] if isinstance(s['prod'], dict) else {}
            for rep in range(1, REPNUM + 1):
                rep_status = prod_statuses.get(rep, STATUS_PENDING)
                if rep_status in (STATUS_PENDING, STATUS_PARTIAL, STATUS_FAILED):
                    # ── Re-check status to avoid race-condition duplicates ──
                    current_rep = check_production(sysname, rep)
                    if current_rep not in (STATUS_PENDING, STATUS_PARTIAL, STATUS_FAILED):
                        print(f"\n⏭️  Skipping {sysname} rep {rep}: status changed to '{current_rep}' (already submitted?)")
                        continue
                    action = f"resume prod rep {rep}" if rep_status == STATUS_PARTIAL else f"equil → prod rep {rep}"
                    print(f"\n📋 Advancing {sysname}: {action}")
                    recovery = _read_recovery(os.path.join(RESULTSPATH, 'production', sysname, f'rep_{rep}'))
                    recovery_log = Path(RESULTSPATH, 'production', sysname, f'rep_{rep}', 'log.txt')
                    if recovery and recovery_log.exists() and 'Particle coordinate is NaN' in recovery_log.read_text(errors='replace')[-20000:]:
                        print(f"  Recovery failed again for {sysname} rep {rep}; leaving for inspection")
                        continue
                    if not dry_run:
                        if _check_gpu_and_wait(wait_for_slot, max_jobs):
                            setup_production(sysname, rep)
                        else:
                            submissions_held += 1
    
    if submissions_held > 0:
        print(f"\n⚠️  {submissions_held} submission(s) held — GPU queue at limit. "
              f"Run again later or increase --max-jobs.")


# ── Retry failed jobs ──────────────────────────────────────────────────────
def _check_gpu_and_wait(wait_for_slot, max_jobs):
    """Check GPU queue and optionally wait for a slot. Returns True if ok to submit."""
    counts = count_gpu_jobs()
    if counts is None:
        return True  # can't query, proceed anyway
    if counts['total'] < max_jobs:
        return True
    if wait_for_slot:
        return wait_for_gpu_slot(max_jobs=max_jobs, timeout=3600)
    else:
        print(f"  ⚠️  GPU queue full ({counts['total']}/{max_jobs}), skipping submission.")
        return False


def retry_failed(targets, dry_run=False, max_jobs=MAX_GPU_JOBS):
    """Retry failed build, equilibration, and production jobs."""
    status = get_all_status(targets)
    
    for entry in targets:
        sysname = get_sysname(entry)
        s = status[sysname]
        
        # Retry failed build
        if s['build'] == STATUS_FAILED:
            print(f"\n🔄 Retrying failed build: {sysname}")
            if not dry_run:
                setup_build(sysname, entry)
        
        # Retry failed or partial equilibration
        if s['equil'] in (STATUS_FAILED, STATUS_PARTIAL):
            label = "partial" if s['equil'] == STATUS_PARTIAL else "failed"
            print(f"\n🔄 Retrying {label} equil: {sysname}")
            if not dry_run and _check_gpu_and_wait(False, max_jobs):
                setup_equilibration(sysname)
        
        # Retry failed or partial production replicates
        if isinstance(s['prod'], dict):
            for rep, st in s['prod'].items():
                if st in (STATUS_FAILED, STATUS_PARTIAL):
                    label = "partial" if st == STATUS_PARTIAL else "failed"
                    print(f"\n🔄 Retrying {label} prod rep {rep}: {sysname}")
                    if not dry_run and _check_gpu_and_wait(False, max_jobs):
                        setup_production(sysname, rep)


# ── Continuous monitor ─────────────────────────────────────────────────────
def watch_loop(interval=600, max_jobs=MAX_GPU_JOBS):
    """Continuously monitor and auto-advance. Ctrl+C to stop."""
    targets = load_targets()
    print(f"🔍 Watching {len(targets)} targets every {interval}s (max {max_jobs} GPU jobs). Ctrl+C to stop.\n")
    
    try:
        while True:
            try:
                status = get_all_status(targets)
                print_status(status)
                
                # Auto-advance (build → equil → production)
                auto_advance(targets, dry_run=False, max_jobs=max_jobs, wait_for_slot=False)
            except Exception as e:
                print(f"\n⚠️  Error in watch cycle: {e}")
                traceback.print_exc()
                print(f"  Continuing in {interval}s...")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n👋 Watch stopped.")


# ── Main CLI ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='GPCR MD Pipeline Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s status                  Show status of all targets
  %(prog)s run                     Auto-advance all ready targets
  %(prog)s run --target ADORA2A_2YDV   Advance specific target
  %(prog)s run --stage equil       Only submit equilibration
  %(prog)s run --dry-run           Preview without submitting
  %(prog)s retry-failed            Retry failed jobs
  %(prog)s watch                   Continuous monitor mode
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # status
    subparsers.add_parser('status', help='Print pipeline status')
    
    # run
    run_parser = subparsers.add_parser('run', help='Auto-advance targets')
    run_parser.add_argument('--stage', choices=['build', 'equil', 'prod'], help='Limit to specific stage')
    run_parser.add_argument('--target', help='Specific target name')
    run_parser.add_argument('--dry-run', action='store_true', help='Preview only')
    run_parser.add_argument('--max-jobs', type=int, default=MAX_GPU_JOBS, help='Max GPU jobs before blocking')
    run_parser.add_argument('--no-wait', action='store_true', help='Skip waiting for GPU slot (fail fast)')
    
    # retry-failed
    retry_parser = subparsers.add_parser('retry-failed', help='Retry failed jobs')
    retry_parser.add_argument('--dry-run', action='store_true', help='Preview only')
    retry_parser.add_argument('--max-jobs', type=int, default=MAX_GPU_JOBS, help='Max GPU jobs before blocking')
    
    # watch
    watch_parser = subparsers.add_parser('watch', help='Continuous monitor')
    watch_parser.add_argument('--interval', type=int, default=300, help='Check interval in seconds')
    watch_parser.add_argument('--max-jobs', type=int, default=MAX_GPU_JOBS, help='Max GPU jobs before blocking')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    targets = load_targets()
    
    if args.command == 'status':
        status = get_all_status(targets)
        print_status(status)
    
    elif args.command == 'run':
        auto_advance(targets, stage=args.stage, target_name=args.target,
                     dry_run=args.dry_run, max_jobs=args.max_jobs,
                     wait_for_slot=not args.no_wait)
        if args.dry_run:
            print("\n[DRY RUN] No jobs were actually submitted.")
    
    elif args.command == 'retry-failed':
        retry_failed(targets, dry_run=args.dry_run, max_jobs=args.max_jobs)
        if args.dry_run:
            print("\n[DRY RUN] No jobs were actually submitted.")
    
    elif args.command == 'watch':
        watch_loop(interval=args.interval, max_jobs=args.max_jobs)


if __name__ == '__main__':
    main()
