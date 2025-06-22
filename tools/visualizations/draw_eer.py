"""
Analyze statistics from some log.json files

Example:
python tools/analysis_tools/draw_eer.py plot_curve --work_dirs /your/files/path/ --model_names baseline_1 baseline_2

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
        nargs='*',   # You could allow to input multiple values, e.g., baseline_1 baseline_2 
        help='make sure your model could do some visualization settings')
    parser_plt.add_argument(
        '--markers',
        type=list,
        default=["o", "v", "^", "s", "p", "d", "*"],
        help='the markers that you want to plot')
    parser_plt.add_argument('--name', default='', type=str, help='title of figure')


def parse_args():
    parser = argparse.ArgumentParser(description='Draw ROC Curvy')
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    args = parser.parse_args()
    return args

def categorize_npy_files(folder_path: str, move_to_end_models: str = None) -> Dict[str, List[Tuple[str, str]]]:

    result = {
        'fpr': [],
        'tpr': [],
        'models': set()
    }
    
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' not exists.")
        return result
    
    #  Name pattern: fpr/tpr + model name + classes + .npy
    #  Example: fpr_resnet50_600.npy
    pattern = re.compile(r'^(fpr|tpr)_([\w.+%-]+)_(?:600|220)\.npy$')
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if not filename.endswith('.npy'):
            continue
        
        match = pattern.match(filename)
        if match:
            category, model_name = match.groups()
            result[category].append((model_name, file_path))
            result['models'].add(model_name)

    models = sorted(result['models'])
    if move_to_end_models:
        valid_special_models = [m for m in move_to_end_models if m in models]

        if valid_special_models:
            normal_models = [m for m in models if m not in valid_special_models]
            models = normal_models + valid_special_models 
    result['models'] = models
    
    model_order = {model: i for i, model in enumerate(models)}
    result['fpr'] = sorted(result['fpr'], key=lambda x: model_order[x[0]])
    result['tpr'] = sorted(result['tpr'], key=lambda x: model_order[x[0]])
    
    return result


def main():
    args = parse_args()

    if args.model_names  == []:
        raise ValueError("Please provide the model names.")
    else:
        print(args.model_names)
    
    categorized_files = categorize_npy_files(args.work_dirs, args.model_names)
    
    print("Classification Results:")
    for category in ['fpr', 'tpr']:
        print(f"\n{category.upper()} Files:")
        for model, path in categorized_files[category]:
            print(f"Model Name: {model}, Files: {os.path.basename(path)}")
    
    print("\nAll Models:")
    for model in categorized_files['models']:
        print(f"  - {model}")

    colors = plt.cm.viridis(np.linspace(0, 1, len(categorized_files['models'])))
    print("The length of model you choosen is  {}".format(len(args.model_names)))
    draw_roc(categorized_files, colors=colors, name=args.name, model_len=len(args.model_names))
    print("Successfully  drawed ROC curve!")

    
def draw_roc(categorized_files, colors, name='none', model_len=1):
    
    # Step 1. need to load the files and divided into FPR and TPR for each backbones.
    # Step 2. need to draw the ROC figure and choose the best results for using the lines
    # Step 3. ensure that ecah backbones with their own legend.

    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(9, 6))
    sns.set_style("whitegrid", rc={'grid.linestyle': '--',
                                   "axes.edgecolor": '.20',
                                   })
    plt.xlim(0.001 * 2.5, 1)
    plt.xscale('log')
    plt.xticks([0.001, 0.01, 0.1, 1])

    plt.ylim(0.0, 1.01)            
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.01])

    for i in range(len(categorized_files['models'])):
        fpr_file, tpr_file = categorized_files['fpr'][i][-1], categorized_files['tpr'][i][-1]
        fpr, tpr = np.load(fpr_file), np.load(tpr_file)
        if i >= len(categorized_files['models']) - int(model_len):
            plt.plot(fpr, tpr, color="lightblue", linewidth=1, linestyle='-', label=categorized_files['models'][i])
        else:
            plt.plot(fpr, tpr, color=colors[i], linewidth=1, linestyle='--', label=categorized_files['models'][i])

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    name = name + '.svg'
    plt.savefig(name, format="svg")
    plt.show()

if __name__ == "__main__":
    main()

