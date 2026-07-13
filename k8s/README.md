# Phase 5 — Kubernetes (local, via kind)

Runs the exact same image and mounted artifacts as `docker-compose.yml`
(Phase 4), just orchestrated by Kubernetes instead of Compose directly —
same CPU-only, correctness-over-speed testing rationale as before. kind was
picked over minikube because it runs Kubernetes nodes as plain Docker
containers, directly on top of the Docker Desktop setup already proven
working in Phase 4 — no separate VM/hypervisor to install or debug.

## Prerequisites

You should already have, from Phase 4:
- Docker Desktop running
- `outputs/{extraction,summarization}-Qwen3-4B/final_adapter/` and
  `data/processed/{reports.jsonl,qdrant_index/}` present in your local clone

New for this phase:
```bash
brew install kind kubectl
```

## 1. Build the image (if you haven't already, or if source changed)

```bash
docker build -t clinical-report-assistant-api:latest -f docker/Dockerfile .
```

## 2. Create the kind cluster

Run this from the **repo root** — `k8s/kind-config.yaml`'s host paths are
relative to wherever this command runs from.

```bash
kind create cluster --name clinical-report-assistant --config k8s/kind-config.yaml
```

If it fails to find `./outputs` or `./data/processed`, edit
`k8s/kind-config.yaml` and replace those two `hostPath` values with the
absolute path from `pwd`.

## 3. Load the image into the cluster

kind's nodes can't see images that only exist in Docker Desktop's local
image cache by default — they need to be explicitly loaded in, since
`imagePullPolicy: Never` in `deployment.yaml` means Kubernetes will never
try to pull this from a registry (there isn't one to pull from).

```bash
kind load docker-image clinical-report-assistant-api:latest --name clinical-report-assistant
```

Re-run this any time you rebuild the image with `docker build`.

## 4. Apply the manifests

```bash
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
```

## 5. Watch it start

Model loading on CPU took 10+ minutes in Phase 4 testing — expect the same
here. `deployment.yaml`'s `startupProbe` is configured to tolerate up to 20
minutes before Kubernetes gives up.

```bash
kubectl get pods --watch
```

Once the pod shows `1/1 Running`, check what actually happened during
startup:

```bash
kubectl logs -f deployment/clinical-report-assistant
```

## 6. Reach the service

```bash
kubectl port-forward service/clinical-report-assistant 8000:8000
```

Then, from another terminal, the exact same requests as Phase 4:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What have we typically found in similar cases of mild cardiomegaly with clear lungs?", "max_new_tokens": 100}'
```

## Cleaning up

```bash
kind delete cluster --name clinical-report-assistant
```

## What hasn't been verified yet

Same disclosure as every other piece of this project's infra: these
manifests were written and reviewed carefully, and their YAML syntax was
checked, but **none of this has actually been run** — there's no Docker/kind
available in the sandbox this project was built in. The hostPath/extraMounts
bridging in particular (two hops: host machine -> kind node -> pod) is the
part most likely to need a small fix on the first real attempt, similar to
how the Docker Compose volume mount needed a `:ro` fix after actually
running it. Treat the first `kind create cluster` + `kubectl apply` the same
way as every other "first real run" in this project: a debugging pass, not
a sure thing.
