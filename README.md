# columnspan-engine

Physical column data and native interchange.

Python 3.11 or newer. The primary interfaces are importable component libraries and a local JSON HTTP service.

## Install

```sh
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements-dev.txt
./.venv/bin/python -m pip install --no-deps -e .
```

## Run

```sh
./.venv/bin/python -m columnspan demo
./.venv/bin/python -m columnspan serve --port 8080
./.venv/bin/python -m pytest -q
```

The server binds only to `127.0.0.1`. `GET /health` reports process readiness; `GET /capabilities` lists operations.
Each component has `POST /v1/<operation>` accepting its JSON request object. Responses contain `result` or an explicit error. The service accepts bodies up to 1 MB and at most four concurrent operation handlers. It does not expose a shell, arbitrary file paths or remote URLs.

## Components

| Operation | Behavior |
|---|---|
| `pages` | Real Parquet int64 column read/write using PyArrow; direct little-endian PLAIN int64 values codec. |
| `ownership` | A compiled C Arrow producer transfers arrays to PyArrow. Slices keep native data alive until their last owner is released. |
| `views` | Bounded exact address enumeration for strided tensor views; overlapping addresses are read-only. |
| `bitmap` | Blocked uint32 sets with union, intersection, difference, inclusive rank and zero-based select. |
| `layout` | Fixed primitive field offsets, natural alignment, arrays and explicit byte order. |

## Request examples

Run `python -m columnspan.examples` to print valid JSON requests. The output is generated locally and contains no credentials.

```sh
./.venv/bin/python -m columnspan.examples > requests.json
curl -s http://127.0.0.1:8080/capabilities
```

Use one operation request from that output as the JSON body for its `/v1/<operation>` endpoint. The `demo` command executes all five operations and prints their computed results.

## Supported scope

The Parquet interface accepts exactly one int64 column named value. It delegates the full container decoder to PyArrow; the direct PLAIN codec operates only on integer value bytes. Tensor analysis is bounded to 100,000 elements. Wire layouts are fixed-size and do not emulate platform C bitfields. Native interchange requires a C compiler and is built into the operating-system temporary directory.

Runtime state uses explicit temporary directories or caller-provided library objects. Service examples are per-request computations; use the library interfaces for persistent state and long-lived readers.

## Dependencies

Pinned Python dependencies are listed in `requirements.txt`; the development test runner is pinned in `requirements-dev.txt`. System libraries or a native compiler, when required above, are detected at runtime. No accounts, network connection strings, tokens or private-key files are required.
