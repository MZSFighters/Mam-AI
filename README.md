# Mam-AI

<!-- Once the necessary software is installed on your computer: To use the RCP Cluster: runai config cluster rcp-caas-->

Log in to runai:
`runai login`

<!--To list the different projects you might have: runai list project-->


To configure a project to work on right now
runai config project light-$GASPAR


To submit a job: runai submit \
  --name meditron-basic \
  --image registry.rcp.epfl.ch/multimeditron/basic:latest-$GASPAR\
  --pvc light-scratch:/mloscratch \
  --large-shm \
  -e NAS_HOME=/mloscratch/users/$GASPAR \
  -e HF_API_KEY_FILE_AT=/mloscratch/users/$GASPAR/keys/hf_key.txt \
  -e WANDB_API_KEY_FILE_AT=/mloscratch/users/$GASPAR/keys/wandb_key.txt \
  -e GITCONFIG_AT=/mloscratch/users/$GASPAR/.gitconfig \
  -e GIT_CREDENTIALS_AT=/mloscratch/users/$GASPAR/.git-credentials \
  -e VSCODE_CONFIG_AT=/mloscratch/users/$GASPAR/.vscode-server \
  --backoff-limit 0 \
  --run-as-gid 84257 \
  --node-pool h100 \
  --gpu 1 \
  -- sleep infinity


  runai describe job meditron-basic

  runai delete job meditron-basic


runai exec mam-rag -it bash


cd /mloscratch/users/moonsamy


python train.py

### To create an environment: 
python -m venv env_name
