"""Run a node: ``python -m edgecluster.serving --config configs/node.example.yaml``."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from edgecluster.config import load_config
from edgecluster.serving.app import create_app


def main() -> None:
    """Parse arguments and serve with uvicorn (single worker: one model copy in RAM)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="YAML config (cluster + node keys)")
    ap.add_argument("--node-id", help="override node_id")
    ap.add_argument("--port", type=int, help="override port")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()
    overrides = {k: v for k, v in {"node_id": args.node_id, "port": args.port}.items() if v is not None}
    cfg = load_config(args.config, **overrides)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level=args.log_level, access_log=False)


if __name__ == "__main__":
    main()
