#!/usr/bin/env bash

if [ -z ${REPO_ROOT+x} ]; then echo "\$REPO_ROOT must be set to the root of nip-deeprl-project git clone."; exit; else cd $REPO_ROOT; fi
if [ -z ${GCR_IMAGE+x} ]; then echo "\$GCR_IMAGE must be set to the name of your docker image on gcr.io."; exit; fi
if [ -z ${GOOGLE_SERVICE_ACCOUNT+x} ]; then echo "\$GOOGLE_SERVICE_ACCOUNT is not set. You can get this from GCE VMs Webinterface -> Create Instance -> (at the very bottom in blue) View this as command-line"; exit; fi
if [ -z ${PROJECT_NAME+x} ]; then echo "\$PROJECT_NAME must be set to the name you gave your project on Google Cloud Platform."; exit; fi
if [ -z ${BUCKET_NAME+x} ]; then echo "\$BUCKET_NAME must be set to the name you gave your bucket on Google Cloud Storage."; exit; fi
ssh-add ~/.ssh/google_compute_engine

zones=( "europe-west1-d" "europe-west2-a" "us-east4-c" "us-east1-b" "us-central1-a" "us-west1-a" "asia-east1-a" "asia-southeast1-a")



# ONLY CHANGE THIS ###########################
exps=(
    'LL_basic'
)
remote=0 # Set to one if docker should be built on the server
#################################################



if [ ${#exps[@]} -gt 64 ]; then echo 'Too many experiments specified at once. Max is 64'; exit; fi

for ((e=0;e<${#exps[@]};++e)); do
    zone=$(($e / 8))
    gcloud_init=(gcloud compute --project "$PROJECT_NAME" \
                        instances create "instance-$e" \
                        --zone "${zones[zone]}" \
                        --machine-type "custom-1-6656" \
                        --subnet "default" \
                        --maintenance-policy "MIGRATE" \
                        --service-account "$GOOGLE_SERVICE_ACCOUNT" \
                        --scopes "https://www.googleapis.com/auth/cloud-platform" \
                        --image "ubuntu-1604-xenial-v20170619a" \
                        --image-project "ubuntu-os-cloud" \
                        --boot-disk-size "50" \
                        --boot-disk-type "pd-standard" \
                        --boot-disk-device-name "instance-$e"
                )
    cron_scp=(gcloud compute scp $REPO_ROOT/gce/rsync-gstorage-bucket instance-$e:~ --zone ${zones[zone]})
    cron_setup=(gcloud compute ssh instance-$e --project $PROJECT_NAME --zone ${zones[zone]} -- \
                       "echo '* * * * *   root    /usr/bin/gsutil -m rsync -r /home/${USER}/data gs://${BUCKET_NAME} > /home/${USER}/CRON.log' > ~/rsync-gstorage-bucket && \
                       sudo chown root:root ~/rsync-gstorage-bucket && \
                       sudo mv ~/rsync-gstorage-bucket /etc/cron.d/"
                )
    docker_scp=(gcloud compute scp --recurse $REPO_ROOT/scripts/ instance-$e:~ --zone ${zones[zone]})
    docker_local=(gcloud compute ssh instance-$e \
                         --project $PROJECT_NAME \
                         --zone ${zones[zone]} -- \
                         "bash ~/scripts/install-docker-ce.sh && \
                         bash ~/scripts/dockercfg-update.sh && \
                         sudo docker pull $GCR_IMAGE && \
                         sudo docker run --rm -d \
                                -v ~/data:/mnt/data $GCR_IMAGE \
                                --capture-videos \
                                --config ${exps[e]}"
                 )
    docker_remote_scp=(gcloud compute scp --recurse $REPO_ROOT instance-$e:~/code --zone ${zones[zone]})
    docker_remote=(gcloud compute ssh instance-$e \
                          --project $PROJECT_NAME \
                          --zone ${zones[zone]} -- \
                          "cd ~/code && \
                          sudo docker build -t testbench -f Dockerfile.cpu . && \
                          sudo docker run --rm -d \
                                  -v ~/data:/mnt/data testbench \
                                  --capture-videos \
                                  --config ${exps[e]}"
                  )
    if [ $remote -eq 0 ]; then
        ${gcloud_init[@]} &
        sleep 1s && "${cron_setup[@]}" && "${docker_scp[@]}" && "${docker_local[@]}"
    else
        ${gcloud_init[@]} &
        sleep 1m && "${cron_setup[@]}" && "${docker_scp[@]}" && "${docker_remote_scp[@]}" && "${docker_remote[@]}" &
    fi
done
