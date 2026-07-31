#!/bin/bash
set -euo pipefail

# Substituted by deploy.py. Values are shell-quoted before insertion.
SOURCE_URI=__SOURCE_URI__
MODEL_ID=__MODEL_ID__
HF_SECRET_ID=__HF_SECRET_ID__
AWS_DEPLOY_REGION=__AWS_REGION__
MAX_MODEL_LEN=__MAX_MODEL_LEN__
SWAP_GIB=__SWAP_GIB__
NEURON_CC_FLAGS_VALUE=__NEURON_CC_FLAGS__

exec > >(tee /var/log/gemma4-jax-inf2-bootstrap.log | logger -t gemma4-inf2 -s 2>/dev/console) 2>&1

export DEBIAN_FRONTEND=noninteractive
apt-get update
# The Neuron DLAMI already ships AWS CLI v2 and the Python the Neuron wheels are
# built against; do not install a second interpreter or a v1 CLI over them.
apt-get install -y python3-venv
command -v aws >/dev/null || { echo "FATAL: AWS CLI missing from the AMI" >&2; exit 1; }

# inf2.xlarge is 4 vCPU / 16 GB host. Field-measured on the sibling NxD port:
# the one-time Neuron graph load peaks ~14.5 GB, and a stock DLAMI with no swap
# OOM-kills the serving process AND the SSM agent -- which also costs you the
# only way back into a host with no inbound SSH. Swap first, install second.
if [ "$SWAP_GIB" -gt 0 ] && ! swapon --show=NAME --noheadings | grep -q '^/swapfile$'; then
  fallocate -l "${SWAP_GIB}G" /swapfile ||
    dd if=/dev/zero of=/swapfile bs=1M count=$((SWAP_GIB * 1024))
  chmod 0600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >>/etc/fstab
fi

CACHE_ROOT=/opt/gemma4/cache
# Owned by ubuntu, not root: the venv below is created as ubuntu *inside* this
# directory, and a root-owned parent fails it with EACCES. Under `set -e` that
# aborts the bootstrap before pip and systemd ever run, leaving a host that
# looks booted and serves nothing.
install -d -o ubuntu -g ubuntu /opt/gemma4

# Find the separate EBS cache volume: on Nitro the attachment point is remapped
# to an NVMe name, so identify it as the one disk the root filesystem is not on.
# inf2 has no instance store, so any other disk is ours.
root_source="$(findmnt -no SOURCE /)"
root_disk="$(lsblk -no PKNAME "$root_source" 2>/dev/null || true)"
[ -n "$root_disk" ] || root_disk="$(basename "$root_source")"
cache_dev=""
while read -r name kind; do
  [ "$kind" = disk ] || continue
  [ "$name" = "$root_disk" ] && continue
  cache_dev="/dev/$name"
  break
done < <(lsblk -dno NAME,TYPE)

if [ -n "$cache_dev" ]; then
  # Only format a blank volume; a reattached one already holds the Neuron and
  # XLA compile caches, which are the whole reason it survives termination.
  blkid "$cache_dev" >/dev/null 2>&1 || mkfs.ext4 -L gemma4cache "$cache_dev"
  install -d "$CACHE_ROOT"
  grep -q '^LABEL=gemma4cache' /etc/fstab ||
    echo "LABEL=gemma4cache $CACHE_ROOT ext4 defaults,nofail 0 2" >>/etc/fstab
  mount "$CACHE_ROOT"
else
  echo "WARNING: no separate cache volume attached; caches land on the root volume" >&2
fi

install -d -o ubuntu -g ubuntu /opt/gemma4/app
install -d -o ubuntu -g ubuntu "$CACHE_ROOT/huggingface"
install -d -o ubuntu -g ubuntu "$CACHE_ROOT/jax"
install -d -o ubuntu -g ubuntu "$CACHE_ROOT/neuron"
chown ubuntu:ubuntu "$CACHE_ROOT"

aws s3 cp "$SOURCE_URI" /tmp/gemma4-source.tar.gz --region "$AWS_DEPLOY_REGION"
tar -xzf /tmp/gemma4-source.tar.gz -C /opt/gemma4/app --strip-components=1
chown -R ubuntu:ubuntu /opt/gemma4/app

sudo -u ubuntu python3 -m venv /opt/gemma4/venv
sudo -u ubuntu /opt/gemma4/venv/bin/python -m pip install --upgrade pip
# Use an SDK-2.28 DLAMI for Inf2. The stable metapackage then selects the tested
# JAX/jaxlib/libneuronxla combination from that Neuron package repository.
# --extra-index-url, NOT --index-url. The Neuron repository carries the Neuron
# wheels only; `jax` itself lives on PyPI. `--index-url` *replaces* the default
# index, so the jax-neuronx dependency `jax<=0.6.2,>=0.4.30` resolves against a
# repository that has never held a jax wheel and the install dies with
# "No matching distribution found for jax" after downloading ~100 MB of
# libneuronxla. `--extra-index-url` is also the form AWS documents.
#
# libneuronxla is pinned SEPARATELY and deliberately. jax-neuronx 0.6.2.1.0
# requires only `libneuronxla>=2.2.12677.0` — an unbounded lower bound — so pip
# resolves the newest, currently 3.0.3854.0. That build targets an NRT 3.0
# runtime, and this AMI line ships NRT 2.31. The install succeeds, and the
# failure surfaces much later as a symbol lookup error at PJRT load:
#
#   libneuronpjrt.so: undefined symbol: nrta_event_register_xu_completion,
#   version NRT_3.0.0
#
# MEASURED on ami-05235a8b272ee7f7e (Neuron SDK 2.29.1, aws-neuronx-runtime-lib
# 2.31.24.0): with libneuronxla pinned to the 2.2 line, jax_neuron/probe.py
# discovers both NeuronCores and executes the decoder block. Move this pin only
# together with the AMI, and re-run the probe when you do.
sudo -u ubuntu /opt/gemma4/venv/bin/python -m pip install \
  'jax-neuronx[stable]==0.6.2.1.0.*' 'libneuronxla==2.2.*' \
  --extra-index-url https://pip.repos.neuron.amazonaws.com
# deployments/aws-inf2/requirements-serving.txt, NOT the repo-root
# requirements.txt — that one lists the MCP server's dependencies and none of
# the serving stack, so a host built from it installs cleanly and then dies on
# `import fastapi`.
sudo -u ubuntu /opt/gemma4/venv/bin/python -m pip install \
  -r /opt/gemma4/app/deployments/aws-inf2/requirements-serving.txt

cat >/usr/local/bin/gemma4-fetch-hf-token <<'SCRIPT'
#!/bin/bash
set -euo pipefail
umask 077
tmp="$(mktemp /run/gemma4-hf-token.XXXXXX)"
trap 'rm -f "$tmp"' EXIT
aws secretsmanager get-secret-value \
  --region "$AWS_DEPLOY_REGION" \
  --secret-id "$HF_SECRET_ID" \
  --query SecretString \
  --output text > "$tmp"
# This runs as root (ExecStartPre=+) but the service runs as ubuntu, so hand the
# file over explicitly; a root-owned 0600 token makes the unit crash-loop.
chown ubuntu:ubuntu "$tmp"
chmod 0400 "$tmp"
mv -f "$tmp" /run/gemma4-hf-token
trap - EXIT
SCRIPT
chmod 0755 /usr/local/bin/gemma4-fetch-hf-token

cat >/usr/local/bin/gemma4-jax-inf2-run <<'SCRIPT'
#!/bin/bash
set -euo pipefail
export HF_TOKEN
HF_TOKEN="$(cat /run/gemma4-hf-token)"
exec /opt/gemma4/venv/bin/python \
  /opt/gemma4/app/deployments/aws-inf2/neuron_entrypoint.py \
  --model "$MODEL_ID" \
  --kv-cache-dtype int8 \
  --quant-mode w4a16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --host 127.0.0.1 \
  --port 8000
SCRIPT
chmod 0755 /usr/local/bin/gemma4-jax-inf2-run

cat >/etc/gemma4-inf2.env <<EOF
# neuronx-cc is a console script in the venv, and libneuronxla shells out to it
# by bare name to compile every graph. systemd hands the unit a default PATH of
# /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin, which
# does not include it, so the first compile dies inside XLA as
#   XlaRuntimeError: UNKNOWN: sh: 1: neuronx-cc: not found
# — a message that points at the compiler rather than at PATH. /opt/aws/neuron/bin
# carries neuron-ls and friends for diagnostics.
PATH=/opt/gemma4/venv/bin:/opt/aws/neuron/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
AWS_DEPLOY_REGION=$AWS_DEPLOY_REGION
HF_SECRET_ID=$HF_SECRET_ID
MODEL_ID=$MODEL_ID
MAX_MODEL_LEN=$MAX_MODEL_LEN
NEURON_CC_FLAGS=$NEURON_CC_FLAGS_VALUE
HF_HOME=$CACHE_ROOT/huggingface
JAX_COMPILATION_CACHE_DIR=$CACHE_ROOT/jax
NEURON_COMPILE_CACHE_URL=$CACHE_ROOT/neuron
EOF
chmod 0600 /etc/gemma4-inf2.env

cat >/etc/systemd/system/gemma4-jax-inf2.service <<'UNIT'
[Unit]
Description=Gemma 4 pure-JAX server on AWS Inferentia2
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
EnvironmentFile=/etc/gemma4-inf2.env
ExecStartPre=+/usr/local/bin/gemma4-fetch-hf-token
ExecStart=/usr/local/bin/gemma4-jax-inf2-run
Restart=on-failure
RestartSec=15
TimeoutStartSec=3600
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now gemma4-jax-inf2.service
