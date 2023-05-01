echo "Running CNN-LSTM-RES in docker"
docker run  --ipc=host --gpus all -it --rm -v $PWD:/app/ patrickdmiller/seed-eeg-deepnn bash -c "cd /app/ && python seed-deep-cnn-runner.py"

