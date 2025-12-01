#!/bin/bash
xhost +local:docker
docker run -it \
    --rm \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd)/training_results:/workspace/training_results \
    -v $(pwd)/models:/workspace/models \
    socnavgym:latest