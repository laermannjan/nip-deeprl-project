#!/usr/bin/env bash

ssh-add ~/.ssh/google_compute_engine

zones=( "europe-west1-d" "europe-west2-a" "us-east4-c" "us-east1-b" )
exps=(  'LL_e1_short_in' 'LL_e1_short_eq' 'LL_e1_short_de' 'LL_e1_long_in' 'LL_e1_long_eq' 'LL_e1_long_de' 'LL_e10_short_in' 'LL_e10_short_eq' 'LL_e10_short_de' 'LL_e10_long_in' 'LL_e10_long_eq' 'LL_e10_long_de' 'AB_e1_short_in' 'AB_e1_short_eq' 'AB_e1_short_de' 'AB_e1_long_in' 'AB_e1_long_eq' 'AB_e1_long_de' 'AB_e10_short_in' 'AB_e10_short_eq' 'AB_e10_short_de' 'AB_e10_long_in' 'AB_e10_long_eq' 'AB_e10_long_de' 'LL_e1_short_eq_prio' 'LL_e1_long_eq_prio' 'LL_e10_short_eq_prio' 'LL_e10_long_eq_prio' 'AB_e1_short_eq_prio' 'AB_e1_long_eq_prio' 'AB_e10_short_eq_prio' 'AB_e10_long_eq_prio' )

# Pull newest docker image
# and 
# Start an experiment
for ((z=0;z<${#zones[@]};++z)); do
    for i in {0..7}; do
        n=$(($z * 8 + $i + 1))
        cmd=(gcloud compute ssh instance-$n \
                    --project nip-deeprl-project \
                    --zone ${zones[z]} -- \
                    /usr/share/google/dockercfg_update.sh \&\& \
                    docker pull eu.gcr.io/nip-deeprl-project/testbench:latest \&\&
                    docker run -d \
                           -v /home/$USER/data:/mnt/data eu.gcr.io/nip-deeprl-project/testbench:latest \
                           --config ${exps[n]} --repeat 2 \
                           --write-upon-reset --capture-videos)
        "${cmd[@]}" &
    done
done
