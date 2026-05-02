import sys


def should_show_progress(progress, stream=None):
    if progress is None:
        output_stream = stream or sys.stderr
        return bool(getattr(output_stream, "isatty", lambda: False)())
    return bool(progress)


def emit_progress(progress_callback, stage, current, total, message=None):
    if progress_callback is None:
        return
    event = {
        "stage": str(stage),
        "current": int(current),
        "total": int(total),
    }
    if message:
        event["message"] = str(message)
    progress_callback(event)


def progress_iter(
    iterable,
    *,
    total=None,
    desc=None,
    unit="item",
    progress=False,
    progress_callback=None,
    stage=None,
):
    progress_stage = stage or desc or "progress"
    progress_total = _resolve_total(iterable, total)
    emit_progress(progress_callback, progress_stage, 0, progress_total, message=desc)

    wrapped = iterable
    progress_bar = None
    if progress:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
        if tqdm is not None:
            progress_bar = tqdm(
                iterable,
                total=progress_total,
                desc=desc,
                unit=unit,
                leave=False,
                file=sys.stderr,
            )
            wrapped = progress_bar

    try:
        for index, item in enumerate(wrapped, start=1):
            yield item
            emit_progress(progress_callback, progress_stage, index, progress_total, message=desc)
    finally:
        if progress_bar is not None:
            progress_bar.close()


def _resolve_total(iterable, total):
    if total is not None:
        return max(0, int(total))
    try:
        return max(0, len(iterable))
    except TypeError:
        return 0
