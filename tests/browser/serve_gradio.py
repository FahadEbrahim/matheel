import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--official-benchmark", type=Path)
    parser.add_argument("--dataset-barriers", type=Path)
    parser.add_argument("--app-source", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, os.fspath(repository_root))
    if args.official_benchmark is not None:
        from gradio_app import official_benchmark

        # Exercise the real validator/view with a tiny frozen fixture, keeping
        # browser tests independent of the published benchmark's size/models.
        official_benchmark.OFFICIAL_BENCHMARK_PATH = args.official_benchmark.resolve()
    app_path = args.app_source or repository_root / "gradio_app" / "app.py"
    spec = importlib.util.spec_from_file_location("matheel_browser_gradio_app", app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Gradio app from {app_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.dataset_barriers is not None:
        barriers = args.dataset_barriers
        barriers.mkdir(parents=True, exist_ok=True)
        for stage, name in (
            ("evaluation", "evaluate_dataset_gradio_for_journey"),
            ("threshold", "threshold_tuning_gradio"),
            ("explanation", "explain_scored_pair_gradio"),
        ):
            original = getattr(module, name)
            def held_callback(*values, _original=original, _stage=stage, **options):
                if not (barriers / f"{_stage}.hold").exists():
                    return _original(*values, **options)
                (barriers / f"{_stage}.started").touch()
                deadline = time.monotonic() + 60
                while not (barriers / f"{_stage}.release").exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Browser callback barrier was not released.")
                    time.sleep(0.02)
                try:
                    return _original(*values, **options)
                finally:
                    (barriers / f"{_stage}.finished").touch()
            setattr(module, name, held_callback)
    module.demo.launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
        quiet=True,
    )


if __name__ == "__main__":
    main()
