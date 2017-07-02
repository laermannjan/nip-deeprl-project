#!/usr/bin/env bash

tstamp=$(date +"%s")
xvfb-run -e /mnt/data/$tstamp.xvfb.err -a -s "-screen 0 800x600x16"\
         -- python $PROJECT_PATH/testbench.py --save-dir /mnt/data $@ > /mnt/data/$tstamp.experiment.log 2>&1
