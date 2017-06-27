#!/usr/bin/env bash

if [ -z ${PROJECT_NAME+x} ]; then echo "\$PROJECT_NAME must be set to the name you gave your project on Google Cloud Platform."; exit; fi
ssh-add ~/.ssh/google_compute_engine

zones=( "europe-west1-d" "europe-west2-a" "us-east4-c" "us-east1-b" "us-central1-a" "us-west1-a" "asia-east1-a" "asia-southeast1-a")
exps=(
    'LL_e1_short_in' 'LL_e1_short_in' 'LL_e1_short_in' 'LL_e1_short_in' 'LL_e1_short_in'
    'LL_e1_short_eq' 'LL_e1_short_eq' 'LL_e1_short_eq' 'LL_e1_short_eq' 'LL_e1_short_eq'
    'LL_e1_short_de' 'LL_e1_short_de' 'LL_e1_short_de' 'LL_e1_short_de' 'LL_e1_short_de'
    'LL_e1_long_in' 'LL_e1_long_in' 'LL_e1_long_in' 'LL_e1_long_in' 'LL_e1_long_in'
    'LL_e1_long_eq' 'LL_e1_long_eq' 'LL_e1_long_eq' 'LL_e1_long_eq' 'LL_e1_long_eq'
    'LL_e1_long_de' 'LL_e1_long_de' 'LL_e1_long_de' 'LL_e1_long_de' 'LL_e1_long_de'
    'LL_e10_short_in' 'LL_e10_short_in' 'LL_e10_short_in' 'LL_e10_short_in' 'LL_e10_short_in'
    'LL_e10_short_eq' 'LL_e10_short_eq' 'LL_e10_short_eq' 'LL_e10_short_eq' 'LL_e10_short_eq'
    'LL_e10_short_de' 'LL_e10_short_de' 'LL_e10_short_de' 'LL_e10_short_de' 'LL_e10_short_de'
    'LL_e10_long_in' 'LL_e10_long_in' 'LL_e10_long_in' 'LL_e10_long_in' 'LL_e10_long_in'
    'LL_e10_long_eq' 'LL_e10_long_eq' 'LL_e10_long_eq' 'LL_e10_long_eq' 'LL_e10_long_eq'
    'LL_e10_long_de' 'LL_e10_long_de' 'LL_e10_long_de' 'LL_e10_long_de' 'LL_e10_long_de'
    'LL_e1_short_eq_prio'
    'LL_e1_long_eq_prio'
    'LL_e10_short_eq_prio'
    'LL_e10_long_eq_prio'
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
                    /usr/share/google/dockercfg_update.sh \&\& \
                    docker pull $GCR_IMAGE \&\&
                    docker run -d \
                           -v /home/$USER/data:/mnt/data $GCR_IMAGE \
                           --config ${exps[n]} --repeat 2 \
                           --write-upon-reset --capture-videos)
        "${cmd[@]}" &
    done
done
