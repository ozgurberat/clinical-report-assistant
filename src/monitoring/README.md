# Phase 6 — monitoring

This directory is intentionally empty. Monitoring ended up small enough
(a `/metrics` endpoint, a request-logging middleware, and a couple of
`prometheus-client` counters/histogram) that it lives directly in
[`src/serving/app.py`](../serving/app.py) instead of as a separate module —
splitting it out here would have meant importing app internals back in,
for no real separation-of-concerns benefit at this size.

See [`src/serving/README.md`](../serving/README.md#monitoring) for what's
actually implemented, and [`k8s/README.md`](../../k8s/README.md#monitoring--scope-decision)
for the scope decision (real `/metrics` + scrape annotations, no live
Prometheus/Grafana deployment).
