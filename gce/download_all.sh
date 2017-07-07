#!/usr/bin/env bash

if [ -z ${PROJECT_NAME+x} ]; then echo "\$PROJECT_NAME must be set to the name you gave your project on Google Cloud Platform."; exit; fi
if [ -z ${REPO_ROOT+x} ]; then echo "\$REPO_ROOT must be set to the root of nip-deeprl-project git clone."; exit; else cd $REPO_ROOT; fi
ssh-add ~/.ssh/google_compute_engine

zones=( "europe-west1-d" "europe-west2-a" "us-east4-c" "us-east1-b" "us-central1-a" "us-west1-a" "asia-east1-a" "asia-southeast1-a")

cd $REPO_ROOT
pids=()
for ((z=0;z<${#zones[@]};++z)); do
    for i in {0..7}; do
        n=$(($z * 8 + $i))
        ssh=(gcloud compute ssh instance-$n\
                    --project $PROJECT_NAME \
                    --zone ${zones[z]} -- \
                    "sudo tar cf data.instance-$n.tar data"
            )
        scp=(gcloud compute scp --recurse instance-$n:~/data.instance-$n.tar $REPO_ROOT --zone ${zones[z]})
        cleanup_remote=(gcloud compute ssh instance-$n\
                        --project $PROJECT_NAME \
                        --zone ${zones[z]} -- \
                        "sudo rm -rf data.instance-$n.tar"
                )
        extract=(tar xvf data.instance-$n.tar)
        cleanup_local=(rm data.instance-$n.tar)

        ${ssh[@]} && ${scp[@]} && ${cleanup_remote[@]} && ${extract[@]} && ${cleanup_local[@]}&
        pids+=("$!")
    done
done

for p in ${pids[*]}; do
    wait $p
done
