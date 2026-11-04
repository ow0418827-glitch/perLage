"""Explicit, read-only integration with the adjacent dual-rag implementation."""
from dataclasses import asdict, replace
from src.paths import ProjectPath as Path
import sys


from src.paths import PROJECT

DEFAULT_DUAL_ROOT = PROJECT / "src/vendor"


def activate(root=DEFAULT_DUAL_ROOT):
    # Reuse the sibling implementation without creating bytecode in its tree.
    sys.dont_write_bytecode = True
    root = Path(root).resolve()
    if not (root / "dual_rag/encoder.py").is_file():
        raise FileNotFoundError(f"dual-rag source not found: {root}")
    source = str(root)
    if source not in sys.path:
        sys.path.insert(0, source)
    import dual_rag
    if not Path(dual_rag.__file__).resolve().is_relative_to(root):
        raise ValueError("Another dual_rag package is already loaded")
    return root


def make_encoder(root=DEFAULT_DUAL_ROOT, optimization_config=None, surrogate=2, device=None):
    root = activate(root)
    from dual_rag.original_attack import resolve_optimization
    from dual_rag.encoder import MatchedClipEncoder
    path = Path(optimization_config).resolve() if optimization_config else PROJECT / "src/configs/optimization.json"
    if surrogate not in (1, 2, 3):
        raise ValueError("surrogate must be 1, 2 or 3")
    config = resolve_optimization(path, surrogate_indices=[surrogate - 1])
    if len(config.surrogates) != 1:
        raise ValueError("Requested surrogate is not in optimization config")
    return MatchedClipEncoder({
        "surrogate": asdict(config.surrogates[0]),
        "surrogate_number": surrogate,
        "device": device or config.device,
    }, root)


def generate_attack(source, target, output_dir, *, root=DEFAULT_DUAL_ROOT,
                    optimization_config=None, method="legacy_pixel", steps=1000,
                    epsilon_levels=16, device=None, target_text=None, text_weight=1.0):
    if method not in ('legacy_pixel', 'lucid'):
        raise ValueError('本桥接入口仅提供 lucid（legacy_pixel）')
    method = 'legacy_pixel'
    if target_text is not None:
        from src.vision.title_alignment import validate_title_options
        validate_title_options(target_text, text_weight, method)
    elif text_weight != 1.0:
        raise ValueError('text_weight requires a target title')
    root = activate(root)
    from dual_rag.original_attack import resolve_optimization
    from dual_rag.optimization.attack_core.engine import run_attack
    config_path = Path(optimization_config).resolve() if optimization_config else PROJECT / "src/configs/optimization.json"
    config = resolve_optimization(config_path)
    if steps < 1 or not 0 < epsilon_levels <= 255:
        raise ValueError("steps and pixel budget must be positive")
    config = replace(config, device=device or config.device,
                     attack=replace(config.attack, epsilon=epsilon_levels / 255,
                                    text_weight=float(text_weight) if target_text is not None else 0.0),
                     acceptance=replace(config.acceptance, decoded_max_pixel_difference=epsilon_levels),
                     runtime=replace(config.runtime, allow_downloads=False))
    if target_text is not None:
        from src.vision.title_alignment import run_title_aligned_attack
        return run_title_aligned_attack(
            source, target, output_dir, config, target_text=target_text,
            steps=steps, run_attack=run_attack)
    result = run_attack(Path(source), Path(target), None, config,
                        method=method, steps=steps, output_directory=Path(output_dir))
    output = Path(result.metadata["output_image"])
    if not output.is_file():
        raise FileNotFoundError(output)
    metadata = result.metadata
    del result
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output, metadata
