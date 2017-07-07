if [ -z ${PROJECT_NAME+x} ]; then echo "\$PROJECT_NAME must be set to the name you gave your project on Google Cloud Platform."; exit; fi
if [ -z ${BUCKET_NAME+x} ]; then echo "\$BUCKET_NAME must be set to the name you gave your bucket on Google Cloud Storage."; exit; fi
ssh-add ~/.ssh/google_compute_engine

zones=( "europe-west1-d" "europe-west2-a" "us-east4-c" "us-east1-b" "us-central1-a" "us-west1-a" "asia-east1-a" "asia-southeast1-a")

for ((z=0;z<${#zones[@]};++z)); do
    for i in {0..7}; do
        n=$(($z * 8 + $i))
        scp=(gcloud compute scp $REPO_ROOT/gce/compress-and-remove-models-remote.sh instance-$n:~ --zone ${zones[z]})
        sync=(docker run -v /home/$USER/data:/mnt/data google/cloud-sdk gsutil -m rsync -r /mnt/data gs://$BUCKET_NAME)
        ssh=(gcloud compute ssh instance-$n\
                              --project $PROJECT_NAME \
                              --zone ${zones[z]} -- \
                              "bash compress-and-remove-models-remote-helper.sh"
            )
        ${scp[@]} && "${ssh[@]}" &
    done
done

