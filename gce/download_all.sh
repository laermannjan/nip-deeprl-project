#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/set-env.sh
if [ -z ${PROJECT_NAME+x} ]; then echo "\$PROJECT_NAME must be set to the name you gave your project on Google Cloud Platform."; exit; fi
if [ -z ${REPO_ROOT+x} ]; then echo "\$REPO_ROOT must be set to the root of nip-deeprl-project git clone."; exit; else cd $REPO_ROOT; fi
if [ -z ${zones+x} ]; then echo "\$zones must be set to define the list of zones you want to use for your Google Compute Engines."; exit; fi
ssh-add ~/.ssh/google_compute_engine


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
        scp=(gcloud compute scp --recurse instance-$n:~/data.instance-$n.tar $REPO_ROOT --zone ${zones[z]} --project $PROJECT_NAME)
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
