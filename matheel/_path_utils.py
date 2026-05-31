from pathlib import Path, PureWindowsPath


def path_name(value):
    text = str(value or "")
    windows_path = PureWindowsPath(text)
    if windows_path.drive or "\\" in text:
        return windows_path.name or text
    return Path(text).name or text


def is_unsafe_relative_path(value):
    text = str(value or "").strip()
    if not text:
        return False
    posix_path = Path(text)
    windows_path = PureWindowsPath(text)
    return (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    )


def relative_path_to_posix(value):
    text = str(value or "").strip()
    windows_path = PureWindowsPath(text)
    if windows_path.drive or "\\" in text:
        parts = windows_path.parts
    else:
        parts = Path(text).parts
    return "/".join(part for part in parts if part not in ("", "."))
