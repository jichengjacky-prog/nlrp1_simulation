#!/usr/bin/env python3
"""Execute a Jupyter notebook unattended: stream output live, save after each cell.

``jupyter nbconvert --execute --inplace`` writes the notebook only once, at the
very end, and does not forward the kernel's stdout to the console.  For a
multi-hour pipeline run that means no visible progress and no partial results if
the kernel dies.

This runner drives the same engine (``nbclient``) but:

* echoes every kernel output message the moment it arrives, so
  ``nohup python utils/run_notebook_nohup.py <notebook> > run.log 2>&1 &`` shows
  progress inside long cells (the notebook's own ``print`` lines included),
* rewrites the notebook after each cell, so partial results survive a crash, and
* keeps going when a cell raises (the error is recorded in the notebook).

Usage::

    python utils/run_notebook_nohup.py notebooks/simulate_orexin_input.ipynb
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def _echo(output) -> None:
    """Write one kernel output object to stdout."""
    if not output:
        return
    kind = output.get('output_type')
    if kind == 'stream':
        sys.stdout.write(output.get('text', ''))
    elif kind in ('execute_result', 'display_data'):
        data = output.get('data', {})
        if 'text/plain' in data:
            sys.stdout.write(str(data['text/plain']) + '\n')
    elif kind == 'error':
        sys.stdout.write('\n'.join(output.get('traceback', [])) + '\n')
    sys.stdout.flush()


class StreamingClient(NotebookClient):
    """NotebookClient that echoes output as it arrives instead of at cell end."""

    def output(self, outs, msg, display_id, cell_index):
        result = super().output(outs, msg, display_id, cell_index)
        _echo(result)
        return result


def main(argv) -> int:
    default = 'notebooks/simulate_orexin_input.ipynb'
    nb_path = Path(argv[1] if len(argv) > 1 else default).resolve()
    if not nb_path.is_file():
        print(f'error: notebook not found: {nb_path}', file=sys.stderr)
        return 2

    nb = nbformat.read(str(nb_path), as_version=4)
    client = StreamingClient(
        nb,
        timeout=None,               # pipeline cells run for hours
        allow_errors=True,          # one bad cell must not stop the batch
        kernel_name='python3',
        startup_timeout=600,
        resources={'metadata': {'path': str(nb_path.parent)}},
    )

    started_all = time.time()
    print(f'=== running {nb_path} at {time.strftime("%Y-%m-%d %H:%M:%S")} ===',
          flush=True)

    with client.setup_kernel():
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != 'code':
                continue
            started = time.time()
            print(f'\n===== cell {index} start '
                  f'(elapsed {time.time() - started_all:.0f}s) =====', flush=True)
            try:
                client.execute_cell(cell, index)
            except Exception as exc:  # noqa: BLE001 - keep the batch alive
                print(f'!! cell {index} raised {type(exc).__name__}: {exc}',
                      flush=True)
            nbformat.write(nb, str(nb_path))
            print(f'\n===== cell {index} done in {time.time() - started:.0f}s =====',
                  flush=True)

    print(f'\n=== ALL CELLS DONE in {time.time() - started_all:.0f}s ===', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
