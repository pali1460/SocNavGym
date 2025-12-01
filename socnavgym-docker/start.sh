#!/bin/bash
set -e

# Clean stale locks
rm -f /tmp/.X*-lock || true
rm -rf /tmp/xvfb-run.* || true

# Start the virtual display
Xvfb :0 -screen 0 1024x768x24 -ac &

export DISPLAY=:0
sleep 0.5

exec "$@"
