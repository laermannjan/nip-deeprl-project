#!/usr/bin/env bash
if [ -z ${PROJECT_NAME+x} ]; then echo "\$PROJECT_NAME must be set to the name you gave your project on Google Cloud Platform."; exit; fi
ssh-add ~/.ssh/google_compute_engine
zones=( "europe-west1-d" "europe-west2-a" "us-east4-c" "us-east1-b" "us-central1-a" "us-west1-a" "asia-east1-a" "asia-southeast1-a")

for ((z=0;z<${#zones[@]};++z)); do
    for i in {0..7}; do
        n=$(($z * 8 + $i))
        stop=(gcloud compute instances stop instance-$n --zone ${zones[z]})
        update=(gcloud beta compute instances set-scopes instance-$n --zone ${zones[z]} --scopes cloud-platform)
        restart=(gcloud compute instances start instance-$n --zone ${zones[z]})

        ${stop[@]} && ${update[@]} && ${start[@]} && echo hi &
    done
done

