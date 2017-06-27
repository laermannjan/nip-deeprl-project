#!/usr/bin/env bash

if [ -z ${PROJECT_NAME+x} ]; then echo "\$PROJECT_NAME must be set to the name you gave your project on Google Cloud Platform."; exit; fi
ssh-add ~/.ssh/google_compute_engine

zones=( "europe-west1-d" "europe-west2-a" "us-east4-c" "us-east1-b" "us-central1-a" "us-west1-a" "asia-east1-a" "asia-southeast1-a")
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
