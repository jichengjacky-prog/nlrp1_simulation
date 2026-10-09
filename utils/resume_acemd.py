"""Resume NVT production with a smaller timestep and unchanged physical time.

Used only for recovery.json mode=checkpoint. Keep the original XTC frame interval;
the input generator scales trajectoryperiod accordingly. The checkpoint load hook
runs before ACEMD constructs its reporters, so progress uses the corrected step.
"""
import importlib
import json
import math
import os
from pathlib import Path


def reconcile_checkpoint(simulation, timestep_fs):
    import numpy as np
    from openmm import unit

    state = simulation.context.getState(getPositions=True, getVelocities=True)
    for values in (state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                   state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)):
        if not np.isfinite(values).all():
            raise ValueError('Checkpoint contains non-finite positions or velocities')
    time_fs = state.getTime().value_in_unit(unit.femtosecond)
    if not math.isfinite(time_fs) or time_fs < 0 or timestep_fs <= 0:
        raise ValueError('Invalid checkpoint time or timestep')
    step = round(time_fs / timestep_fs)
    if not math.isclose(step * timestep_fs, time_fs, abs_tol=1.0, rel_tol=0):
        raise ValueError('Checkpoint time does not align with the new timestep')
    previous_step = simulation.currentStep
    simulation.integrator.setStepSize(timestep_fs * unit.femtosecond)
    simulation.currentStep = step
    print(f'CHECKPOINT_RECOVERY: time={time_fs / 1e6:.6f} ns; '
          f'step={previous_step}->{step}; timestep={timestep_fs} fs', flush=True)


def main():
    import yaml
    os.environ['ACEMD_NONINTERACTIVE'] = '1'
    os.environ['HTMD_NONINTERACTIVE'] = '1'
    cfg = yaml.safe_load(Path('input.yaml').read_text())
    recovery = json.loads(Path('recovery.json').read_text())
    if recovery['mode'] != 'checkpoint' or not cfg.get('restart'):
        raise ValueError('Expected a checkpoint recovery input')
    if cfg.get('barostat') or cfg.get('extforces'):
        raise ValueError('Only unrestrained NVT checkpoint recovery is supported')
    if not Path('restart.chk').is_file():
        raise FileNotFoundError('restart.chk')
    timestep_fs = float(cfg['timestep'])
    previous_hmr = recovery.get('previous_hmr')
    expected_hmr = recovery['previous_timestep_fs'] > 2 if previous_hmr is None else previous_hmr
    if cfg.get('hmr') != expected_hmr:
        raise ValueError('Recovery must preserve checkpoint hydrogen mass repartitioning')
    old_interval = recovery['previous_timestep_fs'] * recovery['previous_trajectoryperiod']
    if not math.isclose(timestep_fs * cfg['trajectoryperiod'], old_interval):
        raise ValueError('Recovery must preserve XTC frame spacing')

    from openmm.app import Simulation
    original_load = Simulation.loadCheckpoint

    def load_checkpoint(simulation, file):
        original_load(simulation, file)
        reconcile_checkpoint(simulation, timestep_fs)

    Simulation.loadCheckpoint = load_checkpoint
    try:
        importlib.import_module('acemd.acemd')._acemd_cli()
    finally:
        Simulation.loadCheckpoint = original_load


if __name__ == '__main__':
    main()
