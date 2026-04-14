#!/usr/bin/env python3
"""Render a Jupyter notebook to PDF/HTML by temporarily stripping NBDev directives.

NBDev uses #|export and #|default_exp directives that confuse quarto.
This script:
1. Creates a temporary copy with directives removed
2. Renders it with quarto
3. Cleans up

Usage:
    render_notebook.py <input.ipynb> <output.pdf>
    render_notebook.py <input.ipynb> <output.html>
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def strip_nbdev_directives(nb_path: Path) -> dict:
    """Load notebook and strip #| directives from code cells."""
    with open(nb_path) as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = cell['source']
            if isinstance(src, list):
                # Remove lines starting with #|
                cell['source'] = [line for line in src if not line.strip().startswith('#|')]
            else:
                lines = src.split('\n')
                cell['source'] = '\n'.join(line for line in lines if not line.strip().startswith('#|'))

    return nb


def main():
    if len(sys.argv) < 3:
        print("Usage: render_notebook.py <input.ipynb> <output.pdf|html>")
        print("Example: render_notebook.py notebook.ipynb output.pdf")
        sys.exit(1)

    input_notebook = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_notebook.exists():
        print(f"Error: Input notebook not found: {input_notebook}")
        sys.exit(1)

    # Determine output format from extension
    output_ext = output_path.suffix.lower()
    if output_ext == '.pdf':
        output_format = 'pdf'
    elif output_ext in ('.html', '.htm'):
        output_format = 'html'
    else:
        print(f"Error: Unsupported output format: {output_ext}")
        print("Supported formats: .pdf, .html")
        sys.exit(1)

    print(f"Rendering {input_notebook} to {output_path}...")

    # Create cleaned copy in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_notebook = Path(tmpdir) / input_notebook.name

        # Create cleaned copy
        cleaned_nb = strip_nbdev_directives(input_notebook)
        with open(temp_notebook, 'w') as f:
            json.dump(cleaned_nb, f, indent=1)

        # Render with quarto
        result = subprocess.run(
            ['quarto', 'render', str(temp_notebook), '--to', output_format],
            capture_output=True,
            text=True,
            cwd=tmpdir
        )

        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            sys.exit(1)

        # Find rendered output
        temp_output = temp_notebook.with_suffix(f'.{output_format}')

        if temp_output.exists():
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(temp_output), str(output_path))
            print(f"Created: {output_path}")
        else:
            print(f"Error: Expected output not found: {temp_output}")
            sys.exit(1)


if __name__ == '__main__':
    main()
