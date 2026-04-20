#!/bin/bash
set -eo pipefail

# Parameter check
if [ -z "${RANK:-}" ]; then
    echo "Error: RANK environment variable not set!"
    exit 1
fi

# Configuration file path (modify according to actual needs)
SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname "$SCRIPT_PATH")
DEFAULT_RAY_HEAD_IP_FILE="$REPO_PATH/ray_utils/ray_head_ip.txt"
DEFAULT_RAY_HEAD_ADDR_FILE="$REPO_PATH/ray_utils/ray_head_addr.txt"
RAY_HEAD_IP_FILE="${RAY_HEAD_IP_FILE:-$DEFAULT_RAY_HEAD_IP_FILE}"
RAY_HEAD_ADDR_FILE="${RAY_HEAD_ADDR_FILE:-$DEFAULT_RAY_HEAD_ADDR_FILE}"
RAY_PORT=${MASTER_PORT:-29500}  # Default port for Ray, can be modified if needed
RAY_PORT_RETRIES=${RAY_PORT_RETRIES:-20}
RAY_WORKER_CONNECT_RETRIES=${RAY_WORKER_CONNECT_RETRIES:-3}

# Ensure Ray temp directory is user-writable under the repo, not /tmp
if [ -z "$RAY_TMPDIR" ]; then
    # USER_NAME=${USER:-$(whoami)}
    export RAY_TMPDIR="$REPO_PATH/.ray_tmp/"
    # export RAY_TMPDIR="$REPO_PATH/.tmp/"
fi
mkdir -p "$RAY_TMPDIR"
export TMPDIR="$RAY_TMPDIR"
mkdir -p "$(dirname "$RAY_HEAD_IP_FILE")" "$(dirname "$RAY_HEAD_ADDR_FILE")"

# Ray object store memory (bytes)
# Can be set via config -> exported as RAY_OBJECT_STORE_MEMORY env var
# Default: 461708984320 (~430GB) for backward compatibility
RAY_MEMORY="${RAY_OBJECT_STORE_MEMORY:-461708984320}"

choose_head_ip() {
    # If a peer host is provided, pick the source IP used to route to that peer.
    if [ -n "${RAY_PEER_HOST:-}" ] && command -v ip >/dev/null 2>&1; then
        local peer_ip route_src
        peer_ip="$RAY_PEER_HOST"
        if ! [[ "$peer_ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            peer_ip="$(getent ahostsv4 "$RAY_PEER_HOST" 2>/dev/null | awk 'NR==1 {print $1}')"
        fi
        if [ -n "$peer_ip" ]; then
            route_src="$(ip route get "$peer_ip" 2>/dev/null | awk '
                {
                    for (i = 1; i <= NF; i++) {
                        if ($i == "src" && (i + 1) <= NF) {
                            print $(i + 1);
                            exit
                        }
                    }
                }
            ')"
            if [ -n "$route_src" ]; then
                echo "$route_src"
                return 0
            fi
        fi
    fi

    # Prefer routable IPv4 addresses and avoid loopback/link-local interfaces.
    local ip
    ip="$(hostname -I 2>/dev/null | tr ' ' '\n' | awk '
        $0 ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ &&
        $0 !~ /^127\./ &&
        $0 !~ /^169\.254\./ &&
        $0 != "" { print; exit }
    ')"
    if [ -n "$ip" ]; then
        echo "$ip"
        return 0
    fi

    # Fallback to any IPv4 from hostname -I.
    ip="$(hostname -I 2>/dev/null | tr ' ' '\n' | awk '
        $0 ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ { print; exit }
    ')"
    if [ -n "$ip" ]; then
        echo "$ip"
        return 0
    fi

    # Final fallback: resolve hostname to IPv4.
    ip="$(getent ahostsv4 "$(hostname -f)" 2>/dev/null | awk 'NR==1 {print $1}')"
    if [ -n "$ip" ]; then
        echo "$ip"
        return 0
    fi

    echo ""
}

# Head node startup logic
if [ "$RANK" -eq 0 ]; then
    # Get a routable local machine IP address.
    IP_ADDRESS="$(choose_head_ip)"
    if [ -z "$IP_ADDRESS" ]; then
        echo "Error: could not determine Ray head IP address."
        exit 1
    fi

    # Start Ray head node
    echo "Starting Ray head node on rank 0, IP: $IP_ADDRESS"
    echo "Ray object store memory: $RAY_MEMORY bytes"
    echo "Requested Ray port: $RAY_PORT (retries: $RAY_PORT_RETRIES)"

    selected_port=""
    for ((i = 0; i < RAY_PORT_RETRIES; i++)); do
        try_port=$((RAY_PORT + i))
        echo "Trying Ray head port: $try_port"
        if ray start --head --node-ip-address="$IP_ADDRESS" --memory="$RAY_MEMORY" --port="$try_port" --temp-dir="$RAY_TMPDIR"; then
            selected_port="$try_port"
            break
        fi
        echo "Ray head failed on port $try_port; trying next port."
        ray stop --force >/dev/null 2>&1 || true
        sleep 1
    done

    if [ -z "$selected_port" ]; then
        echo "Error: failed to start Ray head after $RAY_PORT_RETRIES port attempts starting from $RAY_PORT."
        exit 1
    fi

    # Write address files only after successful head startup.
    echo "$IP_ADDRESS" > "$RAY_HEAD_IP_FILE"
    echo "${IP_ADDRESS}:${selected_port}" > "$RAY_HEAD_ADDR_FILE"
    echo "Head node IP written to $RAY_HEAD_IP_FILE"
    echo "Head node address written to $RAY_HEAD_ADDR_FILE"
else
    # Worker node startup logic
    echo "Waiting for head node address file..."

    HEAD_ADDRESS=""
    # Wait for file to appear (wait up to 360 seconds)
    for i in {1..360}; do
        if [ -s "$RAY_HEAD_ADDR_FILE" ]; then
            HEAD_ADDRESS="$(cat "$RAY_HEAD_ADDR_FILE")"
            if [ -n "$HEAD_ADDRESS" ]; then
                break
            fi
        fi
        sleep 1
    done

    # Backward-compatible fallback if only the IP file exists.
    if [ -z "$HEAD_ADDRESS" ] && [ -s "$RAY_HEAD_IP_FILE" ]; then
        HEAD_IP="$(cat "$RAY_HEAD_IP_FILE")"
        if [ -n "$HEAD_IP" ]; then
            HEAD_ADDRESS="${HEAD_IP}:${RAY_PORT}"
        fi
    fi

    if [ -z "$HEAD_ADDRESS" ]; then
        echo "Error: Could not get head node address from $RAY_HEAD_ADDR_FILE or $RAY_HEAD_IP_FILE"
        exit 1
    fi

    echo "Starting Ray worker node connecting to head at $HEAD_ADDRESS"
    echo "Ray object store memory: $RAY_MEMORY bytes"

    worker_started=0
    for ((attempt = 1; attempt <= RAY_WORKER_CONNECT_RETRIES; attempt++)); do
        if ray start --memory="$RAY_MEMORY" --address="$HEAD_ADDRESS" --temp-dir="$RAY_TMPDIR"; then
            worker_started=1
            break
        fi
        echo "Ray worker failed to join on attempt ${attempt}/${RAY_WORKER_CONNECT_RETRIES}."
        ray stop --force >/dev/null 2>&1 || true
        sleep 5
    done

    if [ "$worker_started" -ne 1 ]; then
        echo "Error: Ray worker failed to join head after $RAY_WORKER_CONNECT_RETRIES attempts."
        exit 1
    fi
fi
