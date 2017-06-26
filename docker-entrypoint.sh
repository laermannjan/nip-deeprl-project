#!/usr/bin/env bash

source activate py35
xvfb-run -e /dev/stderr -a -s "-screen 0 1400x900x24"\
         -- python $PROJECT_PATH/testbench.py --save-dir /mnt/data $@
