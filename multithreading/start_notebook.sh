#!/bin/bash

set -euo pipefail

PORT=${1-1729}

# This is needed only on eX3, where we need to create a reverse tunnel from the
# compute node to the login node.
if [[ "${HOSTNAME}" == "ipu-pod64-server1" ]]; then
    function cleanup() {
        echo "Killing process with ID ${PROXY_PID}..."
        kill "${PROXY_PID}"
    }

   REMOTE=srl-login1

   # Forward the port back to the login node.
   ssh -N -R "${PORT}:localhost:${PORT}" "${REMOTE}" &
   # Remember the PID
   PROXY_PID=$!
   # At the end of this script, terminate the SSH forwarding.
   trap cleanup EXIT
fi

if [[ "${HOSTNAME}" == "ipu-pod64-server1" ]]; then
    SERVER="dnat.simula.no"
elif [[ "${HOSTNAME}" == *".rc.ucl.ac.uk" ]]; then
    # Fibonacci or Mandelbrot
    SERVER="${HOSTNAME}"
    export JULIAUP_CHANNEL=1.11 # Make sure we use the right channel
    export JULIA_NUM_THREADS=16
fi

echo "On your local machine run
    ssh -f -N ${SERVER} -L ${PORT}:localhost:${PORT}
"

# This was necessary only on eX3 only for a couple of days.  It shouldn't be
# necessary anymore, but we keep it for good measure.
if [[ "${HOSTNAME}" == "ipu-pod64-server1" ]]; then
    # See https://cwrap.org/ and
    # https://developers.redhat.com/blog/2015/05/05/testing-your-software-stack-without-root-privileges-using-cwrap
    RESOLV_WRAPPER_CONF="./hosts"
    export LD_PRELOAD=$(realpath ~/repo/resolv_wrapper/build/src/libresolv_wrapper.so)
fi

julia --project -e "
import Pkg
Pkg.instantiate()
import Pluto
Pluto.run(; launch_browser=false, port=${PORT}, auto_reload_from_file=true)
"
