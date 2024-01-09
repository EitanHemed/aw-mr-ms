#!/bin/bash
# The --login ensures the bash configuration is loaded,
# enabling Conda.

# Enable strict mode.
set -euo pipefail
# ... Run whatever commands ...

# Temporarily disable strict mode and activate conda:
set +euo pipefail
source activate aw_mr  # Use source instead of conda activate

mv reproduc-nb.ipynb "aw-mr-ms/Data Analysis" 

cd "aw-mr-ms/Data Analysis"	

pip install -e .

# Re-enable strict mode:
set -euo pipefail
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token=''

