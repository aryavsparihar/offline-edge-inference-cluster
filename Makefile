# Common tasks. Override variables on the command line, e.g.
#   make compress CONFIG=configs/compress/mbert-bbca.yaml
#   make bench MODEL_DIR=runs/xlmr-bbca/onnx MODEL_FILE=model.pruned.int8.onnx

PYTHON      ?= .venv/bin/python
CONFIG      ?= configs/compress/xlmr-bbca.yaml
RUN         ?= runs/xlmr-bbca
MODEL_DIR   ?= $(RUN)/onnx
MODEL_FILE  ?= model.pruned.int8.onnx
NODE_CONFIG ?= configs/node.local.yaml
NODES       ?= 12
DURATION    ?= 60
GEN_DIR     ?=

.PHONY: install install-runtime test test-fast lint format compress compress-tiny serve sim sweep cluster-demo \
        bench docker-build compose cluster-up cluster-down generator

install:            ## full dev environment (training + serving + tests)
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements/dev.txt
	$(PYTHON) -m pip install --no-deps -e .

install-runtime:    ## edge node only (what runs on the Pi)
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements/runtime.txt
	$(PYTHON) -m pip install --no-deps -e .

test:               ## full suite, including the tiny-model compression end-to-end test
	$(PYTHON) -m pytest -q

test-fast:          ## skip the compression e2e test
	$(PYTHON) -m pytest -q -m "not slow"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

compress:           ## fine-tune, prune, quantize, export and report (CONFIG=...)
	$(PYTHON) -m edgecluster.compression.pipeline --config $(CONFIG)

compress-tiny:      ## the same pipeline on a tiny random model (minutes, CPU)
	$(PYTHON) -m edgecluster.compression.pipeline --config configs/compress/tiny-sample.yaml

serve:              ## run one node (NODE_CONFIG=...)
	$(PYTHON) -m edgecluster.serving --config $(NODE_CONFIG)

sim:                ## in-process 12-node simulation with the default fault settings (one seed)
	$(PYTHON) -m sim.simulator --nodes $(NODES) --json results/sim/default.json

sweep:              ## convergence sweep over loss / partitions / rounds / restarts (writes results/sim/headline.json)
	$(PYTHON) -m sim.sweep --seeds 10 --out results/sim

cluster-demo:       ## 3 real node processes on localhost: partition, heal, converge
	$(PYTHON) -m sim.cluster_demo --nodes 3

bench:              ## latency benchmark under continuous load (run this on the Pi)
	$(PYTHON) -m bench.latency --model-dir $(MODEL_DIR) --model-file $(MODEL_FILE) --duration $(DURATION) \
		--report $(RUN)/report.json $(if $(GEN_DIR),--generator-dir $(GEN_DIR),)

generator:          ## download + smoke-test the optional generative model
	$(PYTHON) -m edgecluster.serving.generator --dest models/qwen2.5-0.5b-instruct

docker-build:
	docker build -f deploy/Dockerfile -t edgecluster:latest .

compose:
	$(PYTHON) -m sim.gen_compose --nodes $(NODES)

cluster-up: docker-build compose  ## N-node docker-compose cluster (default 12)
	docker compose -f sim/docker-compose.yml up -d
	$(PYTHON) -m sim.watch --nodes $(NODES) --until-converged --timeout 120

cluster-down:
	docker compose -f sim/docker-compose.yml down -v
