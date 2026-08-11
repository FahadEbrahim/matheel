import argparse
import importlib.util
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, os.fspath(repository_root))
    app_path = repository_root / "gradio_app" / "app.py"
    spec = importlib.util.spec_from_file_location("matheel_browser_gradio_app", app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Gradio app from {app_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.demo.launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
        quiet=True,
    )


if __name__ == "__main__":
    main()
