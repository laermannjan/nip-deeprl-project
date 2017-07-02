#!/usr/bin/env bash

tstamp=$(date +"%s")
xvfb-run -e /mnt/data/$tstamp.xvfb.err -a -s "-screen 0 1400x900x24"\
         -- python $PROJECT_PATH/testbench.py --save-dir /mnt/data --write-upon-reset $@ > /mnt/data/$tstamp.experiment.log 2>&1
