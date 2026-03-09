"""
Draw ROC curves from fpr/tpr npy files (EER evaluation).

Supports:
- Flat dir: fpr_<model>_600.npy / tpr_<model>_600.npy under --work_dirs.
- Recursive: fpr_None.npy / tpr_None.npy in subdirs; model name inferred from path.

Example:
  python tools/visualizations/draw_eer.py plot_curve --work_dirs /path/to/eer/tju600 --name roc_tju600
  python tools/visualizations/draw_eer.py plot_curve --work_dirs /path/to/eer/tju600 --name roc --smooth   # smooth curves + shaded band
  python tools/visualizations/draw_eer.py plot_curve --work_dirs /path/to/eer/tju600 --model_names swin_b_starmix_sz224_bs32 --name roc
"""

import argparse
import os
import re
from typing import List, Dict, Set, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve

def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        '--work_dirs',
        type=str,
        help='path of fpr & tpr')
    parser_plt.add_argument(
        '--model_names',
        type=str,
        nargs='*',
        default=None,
        help='optional: model names for legend order and to highlight (red) the last N. If omitted, all discovered models are used.')
    parser_plt.add_argument(
        '--markers',
        type=list,
        default=["o", "v", "^", "s", "p", "d", "*"],
        help='the markers that you want to plot')
    parser_plt.add_argument('--name', default='', type=str, help='title of figure')
    parser_plt.add_argument(
        '--smooth',
        action='store_true',
        help='smooth ROC curves with moving average and draw shaded band (like epoch_vs_acc)')
    parser_plt.add_argument(
        '--window_size',
        type=int,
        default=15,
        help='moving average window size when --smooth (default: 15)')


def parse_args():
    parser = argparse.ArgumentParser(description='Draw ROC Curvy')
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    args = parser.parse_args()
    return args

def _infer_model_name_from_path(work_dir: str, npy_file_path: str) -> str:
    """Infer model name from directory path when filename is fpr_None.npy / tpr_None.npy.
    Uses relative path from work_dir to the config dir (parent of dir containing npy).
    E.g. work_dir/tju600, file=.../swin/swin_b_starmix_sz224_bs32/epoch_600/fpr_None.npy
    -> config dir = .../swin/swin_b_starmix_sz224_bs32 -> rel = swin/swin_b_starmix_sz224_bs32
    """
    dir_containing_npy = os.path.dirname(npy_file_path)
    config_dir = os.path.dirname(dir_containing_npy)  # e.g. .../swin_b_starmix_sz224_bs32
    try:
        rel = os.path.relpath(config_dir, work_dir)
    except ValueError:
        rel = config_dir
    # Normalize to a clean label: keep one level if same name (backbone/config) or use rel
    if os.path.sep in rel:
        # e.g. swin/swin_b_starmix_sz224_bs32 -> use last part for short label
        rel = rel.replace(os.path.sep, '/')
    return rel


def categorize_npy_files(folder_path: str, move_to_end_models: List[str] = None) -> Dict[str, List[Tuple[str, str]]]:
    """Collect fpr/tpr npy file pairs and model names.
    Supports two layouts:
    1) Flat: fpr_<model>_600.npy / tpr_<model>_600.npy in folder_path.
    2) Recursive: fpr_None.npy / tpr_None.npy in subdirs; model name inferred from path.
    """
    result = {
        'fpr': [],
        'tpr': [],
        'models': set()
    }
    move_to_end_models = move_to_end_models or []

    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' not exists.")
        return result

    folder_path = os.path.abspath(folder_path)
    # Pattern 1: flat naming fpr_<name>_(600|220).npy
    flat_pattern = re.compile(r'^(fpr|tpr)_([\w.+%-]+)_(?:600|220)\.npy$')
    flat_fpr, flat_tpr = {}, {}
    for filename in os.listdir(folder_path):
        if not filename.endswith('.npy'):
            continue
        match = flat_pattern.match(filename)
        if match:
            category, model_name = match.groups()
            file_path = os.path.join(folder_path, filename)
            if category == 'fpr':
                flat_fpr[model_name] = file_path
            else:
                flat_tpr[model_name] = file_path

    if flat_fpr and flat_tpr:
        for model_name in flat_fpr:
            if model_name in flat_tpr:
                result['fpr'].append((model_name, flat_fpr[model_name]))
                result['tpr'].append((model_name, flat_tpr[model_name]))
                result['models'].add(model_name)
    else:
        # Pattern 2: recursive fpr_*.npy / tpr_*.npy (e.g. fpr_None.npy), infer name from path
        pair_by_key = {}  # (dirpath, suffix) -> (fpr_path, tpr_path)
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if not filename.endswith('.npy'):
                    continue
                m = re.match(r'^fpr_(.+)\.npy$', filename)
                if m:
                    suffix = m.group(1)
                    tpr_name = f'tpr_{suffix}.npy'
                    if tpr_name in files:
                        fpr_path = os.path.join(root, filename)
                        tpr_path = os.path.join(root, tpr_name)
                        model_name = suffix if suffix.lower() != 'none' else _infer_model_name_from_path(folder_path, fpr_path)
                        pair_by_key[(root, suffix)] = (model_name, fpr_path, tpr_path)
                        continue
                m = re.match(r'^tpr_(.+)\.npy$', filename)
                if m:
                    suffix = m.group(1)
                    fpr_name = f'fpr_{suffix}.npy'
                    if fpr_name in files:
                        tpr_path = os.path.join(root, filename)
                        fpr_path = os.path.join(root, fpr_name)
                        model_name = suffix if suffix.lower() != 'none' else _infer_model_name_from_path(folder_path, fpr_path)
                        pair_by_key[(root, suffix)] = (model_name, fpr_path, tpr_path)

        for (_, _), (model_name, fpr_path, tpr_path) in pair_by_key.items():
            result['fpr'].append((model_name, fpr_path))
            result['tpr'].append((model_name, tpr_path))
            result['models'].add(model_name)

    if not result['models']:
        print(f"Warning: No fpr/tpr npy pairs found under '{folder_path}'.")
        return result

    models = sorted(result['models'])
    if move_to_end_models:
        valid_special = [m for m in move_to_end_models if m in models]
        if valid_special:
            normal_models = [m for m in models if m not in valid_special]
            models = normal_models + valid_special
    result['models'] = models

    model_order = {model: i for i, model in enumerate(models)}
    result['fpr'] = sorted(result['fpr'], key=lambda x: model_order.get(x[0], 999))
    result['tpr'] = sorted(result['tpr'], key=lambda x: model_order.get(x[0], 999))
    return result


def _smooth_roc_and_band(fpr: np.ndarray, tpr: np.ndarray, window_size: int = 15, std_multiplier: float = 0.1):
    """Smooth FPR/TPR with moving average and compute TPR band for fill_between.
    Same idea as epoch_vs_acc: convolve with uniform kernel, then band = smoothed ± std_multiplier * std.
    """
    if len(fpr) < window_size or len(tpr) < window_size:
        return fpr, tpr, None, None
    kernel = np.ones(window_size) / window_size
    fpr_ma = np.convolve(fpr, kernel, mode='valid')
    tpr_ma = np.convolve(tpr, kernel, mode='valid')
    std = np.std(tpr_ma)
    upper = tpr_ma + std_multiplier * std
    lower = tpr_ma - std_multiplier * std
    lower = np.clip(lower, 0.0, 1.0)
    upper = np.clip(upper, 0.0, 1.0)
    return fpr_ma, tpr_ma, lower, upper


def main():
    args = parse_args()

    # model_names optional: when None or [], auto-discover all; when given, used for order + highlight
    model_names = args.model_names if args.model_names else []
    if model_names:
        print("Model names (for order & highlight):", model_names)

    categorized_files = categorize_npy_files(args.work_dirs, move_to_end_models=model_names)

    if not categorized_files['models']:
        raise ValueError("No fpr/tpr npy pairs found. Check --work_dirs.")

    print("Classification Results:")
    for category in ['fpr', 'tpr']:
        print(f"\n{category.upper()} Files:")
        for model, path in categorized_files[category]:
            print(f"  {model} -> {os.path.basename(path)}")
    print("\nAll Models (order):")
    for model in categorized_files['models']:
        print(f"  - {model}")

    # colors = plt.cm.viridis(np.linspace(0, 1, len(categorized_files['models'])))
    colors = plt.cm.get_cmap("rainbow", len(categorized_files['models']) + 1)
    model_len = len(model_names) if model_names else 0  # last N models drawn in red when > 0
    print("Highlight (red) last N models: {}".format(model_len if model_len else "none"))
    smooth = getattr(args, 'smooth', False)
    window_size = getattr(args, 'window_size', 15)
    draw_roc(categorized_files, colors=colors, name=args.name, model_len=model_len,
             smooth=smooth, window_size=window_size)
    print("Successfully drew ROC curve!")

    
def draw_roc(categorized_files, colors, name='none', model_len=1, smooth=False, window_size=15):
    # Step 1. Load FPR/TPR per model.
    # Step 2. Optionally smooth with moving average and draw shaded band.
    # Step 3. Draw ROC and legend.

    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(6, 4))
    sns.set_style("whitegrid", rc={
        'grid.linestyle': '--',
        "axes.edgecolor": '.20',
        "axes.spines.right": False,
        "axes.spines.top": False,
    })
    plt.xlim(0.001 * 2.5, 1)
    plt.xscale('log')
    plt.xticks([0.001, 0.01, 0.1, 1])
    plt.ylim(0.8, 1.01)
    plt.yticks([0.8, 0.85, 0.9, 0.95, 1.01])
    # plt.ylim(0.94, 1.01)
    # plt.yticks([0.94, 0.955, 0.97, 0.985, 1.01])

    alpha_band = 0.1

    for i in range(len(categorized_files['models'])):
        fpr_file, tpr_file = categorized_files['fpr'][i][-1], categorized_files['tpr'][i][-1]
        fpr, tpr = np.load(fpr_file), np.load(tpr_file)
        is_highlight = i >= len(categorized_files['models']) - int(model_len)
        color = colors(i)
        linestyle = '-' if is_highlight else '-.'
        linewidth = 1.5 if is_highlight else 1.0
        label = categorized_files['models'][i]

        if smooth and len(fpr) >= window_size and len(tpr) >= window_size:
            fpr_ma, tpr_ma, lower, upper = _smooth_roc_and_band(fpr, tpr, window_size=window_size)
            plt.fill_between(fpr_ma, lower, upper, alpha=alpha_band, color=color, linewidth=0.6)
            plt.plot(fpr_ma, tpr_ma, linestyle=linestyle, color=color, label=label, linewidth=linewidth)
        else:
            plt.plot(fpr, tpr, linestyle=linestyle, color=color, label=label, linewidth=linewidth)

    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, zorder=0, label='Max (TPR=1.0)')
    plt.xlabel('FPR', fontsize=16)
    plt.ylabel('TPR', fontsize=16)
    plt.legend(fontsize=10, loc='lower right', ncol=2, columnspacing=1., handlelength=1.5)
    svg_name = name + '.svg'
    plt.savefig(svg_name, format="svg")
    png_name = name + '.png'
    plt.savefig(png_name, format="png")
    plt.show()

if __name__ == "__main__":
    main()

