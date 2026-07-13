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

## Monitoring — scope decision

Phase 6 added a real `/metrics` endpoint (Prometheus format — request
counts, latency histogram, tokens generated) and structured request logging
to the app itself (see `src/serving/README.md`'s Monitoring section), and
`deployment.yaml` carries the standard `prometheus.io/scrape` annotations
documenting how a Prometheus server would auto-discover this pod. What was
deliberately **not** built: an actual Prometheus + Grafana deployment in
this cluster. That was a scoping call, not an oversight — the skill being
demonstrated is knowing what to instrument and why, which the `/metrics`
endpoint itself proves; standing up the full observability stack would have
meant another round of the same deploy-debug-wait-10-minutes cycle already
gone through twice for Docker and Kubernetes, better spent on the ML side of
this project. If you want the full dashboard later, the annotations and
metrics format are already there to scrape.

## What's been verified vs. what hasn't

`kubectl apply` + a real pod running `1/1 Ready` + all endpoints returning
correct responses through `kubectl port-forward` — all confirmed working, on
the first real attempt, no fixes needed (unlike Docker Compose's `:ro` fix
in Phase 4). What has **not** been verified: the `/metrics` endpoint itself
and the `prometheus.io/scrape` annotations, added after the last real test
run — same disclosure as everything else in this project that's new and
untested. If you want to confirm `/metrics` works on this cluster too, it's
one more `docker build` + `kind load docker-image` + `kubectl rollout
restart deployment/clinical-report-assistant` away.
