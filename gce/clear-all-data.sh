#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/set-env.sh
if [ -z ${PROJECT_NAME+x} ]; then echo "\$PROJECT_NAME must be set to the name you gave your project on Google Cloud Platform."; exit; fi
if [ -z ${zones+x} ]; then echo "\$zones must be set to define the list of zones you want to use for your Google Compute Engines."; exit; fi
ssh-add ~/.ssh/google_compute_engine

for ((z=0;z<${#zones[@]};++z)); do
    for i in {0..7}; do
        n=$(($z * 8 + $i))
        cmd=(gcloud compute ssh instance-$n \
                    --project $PROJECT_NAME \
                    --zone ${zones[z]} -- \
                    sudo rm -rf /home/$USER/data)
        "${cmd[@]}" &
    done
done
