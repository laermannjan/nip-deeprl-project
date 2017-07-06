#!/usr/bin/env bash

if [ -z ${PROJECT_NAME+x} ]; then echo "\$PROJECT_NAME must be set to the name you gave your project on Google Cloud Platform."; exit; fi
ssh-add ~/.ssh/google_compute_engine

zones=( "europe-west1-d" "europe-west2-a" "us-east4-c" "us-east1-b" "us-central1-a" "us-west1-a" "asia-east1-a" "asia-southeast1-a")
exps=(
    'LL_basic' 'LL_basic' 'LL_basic' 'LL_basic' 'LL_basic' 'LL_basic' 'LL_basic' 'LL_basic' 'LL_basic'
    'LL_e500' 'LL_e500' 'LL_e500' 'LL_e500' 'LL_e500' 'LL_e500' 'LL_e500' 'LL_e500' 'LL_e500'
    'LL_rpb250' 'LL_rpb250' 'LL_rpb250' 'LL_rpb250' 'LL_rpb250' 'LL_rpb250' 'LL_rpb250' 'LL_rpb250' 'LL_rpb250'
    'LL_rpb500' 'LL_rpb500' 'LL_rpb500' 'LL_rpb500' 'LL_rpb500' 'LL_rpb500' 'LL_rpb500' 'LL_rpb500' 'LL_rpb500'
    'LL_gc10' 'LL_gc10' 'LL_gc10' 'LL_gc10' 'LL_gc10' 'LL_gc10' 'LL_gc10' 'LL_gc10' 'LL_gc10'
    'LL_gc5' 'LL_gc5' 'LL_gc5' 'LL_gc5' 'LL_gc5' 'LL_gc5' 'LL_gc5' 'LL_gc5' 'LL_gc5'
    'LL_prio1' 'LL_prio1' 'LL_prio1' 'LL_prio1' 'LL_prio1' 'LL_prio1' 'LL_prio1' 'LL_prio1' 'LL_prio1'
)
# Pull newest docker image
# and 
# Start an experiment
for ((z=0;z<${#zones[@]};++z)); do
    for i in {0..7}; do
        n=$(($z * 8 + $i))
        cmd=(gcloud compute ssh instance-$n \
                    --project $PROJECT_NAME \
                    --zone ${zones[z]} -- \
                    "/usr/share/google/dockercfg_update.sh \&\& \
                    docker pull $GCR_IMAGE \&\&
                    docker run -d \
                           -v /home/$USER/data:/mnt/data $GCR_IMAGE \
                           --config ${exps[n]} \
                           --write-upon-reset --capture-videos")
        "${cmd[@]}" &
    done
done
