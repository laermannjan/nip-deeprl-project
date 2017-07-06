#!/usr/bin/env bash

if [ -z ${REPO_ROOT+x} ]; then echo "\$REPO_ROOT must be set to the root of nip-deeprl-project git clone."; exit; else cd $REPO_ROOT; fi
if [ -z ${GCR_IMAGE+x} ]; then echo "\$GCR_IMAGE must be set to the name of your docker image on gcr.io."; exit; fi
if [ -z ${GOOGLE_SERVICE_ACCOUNT+x} ]; then echo "\$GOOGLE_SERVICE_ACCOUNT is not set. You can get this from GCE VMs Webinterface -> Create Instance -> (at the very bottom in blue) View this as command-line"; exit; fi
if [ -z ${PROJECT_NAME+x} ]; then echo "\$PROJECT_NAME must be set to the name you gave your project on Google Cloud Platform."; exit; fi
ssh-add ~/.ssh/google_compute_engine

zones=( "europe-west1-d" "europe-west2-a" "us-east4-c" "us-east1-b" "us-central1-a" "us-west1-a" "asia-east1-a" "asia-southeast1-a")

for ((z=0;z<${#zones[@]};++z)); do
    for i in {0..7}; do
        n=$(($z * 8 + $i))
        cmd=(gcloud compute --project "$PROJECT_NAME" \
                    instances create "instance-$n" \
                    --zone "${zones[z]}" \
                    --machine-type "custom-1-6656" \
                    --subnet "default" \
                    --scopes "https://www.googleapis.com/auth/cloud-platform" \
                    --metadata "^#&&#^google-container-manifest={\u000a  \"apiVersion\": \"v1\",\u000a  \"kind\": \"Pod\",\u000a  \"metadata\": {\u000a    \"name\": \"instance-$n\"\u000a  },\u000a  \"spec\": {\u000a    \"containers\": [\u000a      {\u000a        \"name\": \"instance-$n\",\u000a        \"image\": \"$GCR_IMAGE\",\u000a        \"imagePullPolicy\": \"Always\"\u000a      }\u000a    ]\u000a  }\u000a}#&&#user-data=#cloud-config\u000aruncmd:\u000a- [ '/usr/bin/kubelet', '--allow-privileged=false', '--manifest-url=http://metadata.google.internal/computeMetadata/v1/instance/attributes/google-container-manifest', '--manifest-url-header=Metadata-Flavor:Google' ]#&&#gci-ensure-gke-docker=true" --maintenance-policy "MIGRATE" --service-account "$GOOGLE_SERVICE_ACCOUNT" --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring.write","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --image "cos-stable-59-9460-64-0" --image-project "cos-cloud" --boot-disk-size "50" --boot-disk-type "pd-standard" --boot-disk-device-name "instance-$n")
        "${cmd[@]}" &
    done
done
