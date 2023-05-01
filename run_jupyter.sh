docker run  --ipc=host --gpus all -it --rm -p 8888:8888 -v $PWD:/app/ patrickdmiller/seed-eeg-deepnn  bash -c "cd /app/ && jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 --no-browser --NotebookApp.allow_origin='*'"

