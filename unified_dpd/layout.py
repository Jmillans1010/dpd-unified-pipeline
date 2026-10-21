"""Run directories and read-only compatibility with previously saved runs."""
from pathlib import Path


RUN_DIRECTORIES = ("diagnostics", "model_results", "model_comparisons", "outputs")


def create_run_directories(run):
    """Create the same four categories for full, preprocessing, and feedback runs."""
    for name in RUN_DIRECTORIES:
        (Path(run) / name).mkdir()
    return {name: name for name in RUN_DIRECTORIES}


def resolve_saved_models(source):
    """Accept a run root, its model folder, or an original framework run."""
    source = Path(source).resolve()
    if source.name in ("model_results", "models") and (source.parent / "manifest.json").is_file():
        return source, source.parent / "manifest.json"
    manifest = source / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"No manifest.json found for saved models: {source}")
    for name in ("model_results", "models"):
        if (source / name).is_dir():
            return source / name, manifest
    return source, manifest


def resolve_saved_export(run, model):
    """Read current or legacy exports without moving or rewriting prior results."""
    model = "no_dpd" if model == "sin_dpd" else model
    legacy_name = "sin_dpd" if model == "no_dpd" else model
    candidates = (Path(run) / "outputs" / model, Path(run) / "exports" / legacy_name)
    for directory in candidates:
        if (directory / "metadata.json").is_file() and (directory / "samples.csv").is_file():
            return directory
    raise FileNotFoundError(f"No saved export found for {model}: {run}")
