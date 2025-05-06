import pickle as pkl
import numpy as np
import os
import yaml
import itertools

def generate_latex_table(mean_data, std_data, metric_name, architectures, sample_sizes):
    """Generate LaTeX table for results.
    
    Args:
        mean_data: np.array of shape (num_arch, sample_size)
        std_data: np.array of shape (num_arch, sample_size)
        metric_name: Name of the metric (e.g., "GNN Test Accuracy")
        architectures: List of architecture names/numbers
        sample_sizes: List of sample sizes
    """
    num_arch = len(architectures)
    num_sizes = len(sample_sizes)
    
    latex_code = f"""\\begin{{table}}[h]
\\centering
\\caption{{{metric_name} (Mean $\\pm$ Std) Across Architectures and Sample Sizes}}
\\label{{tab:{metric_name.lower().replace(" ", "_")}}}
\\footnotesize
\\begin{{tabular}}{{|c|*{{{num_arch}}}{{c|}}}}
\\hline
& \\multicolumn{{{num_arch}}}{{|c|}}{{Architecture}} \\\\ 
\\hline
"""
    
    # Add architecture headers
    latex_code += "& " + " & ".join([f"Arch {i+1}" for i in range(num_arch)]) + " \\\\ \n\\hline\n"
    
    # Add data rows
    for size_idx, size in enumerate(sample_sizes):
        row = f"{size}"
        for arch_idx in range(num_arch):
            mean_val = mean_data[arch_idx, size_idx]
            std_val = std_data[arch_idx, size_idx]
            row += f" & ${mean_val:.1f} \\pm {std_val:.1f}$"
        row += " \\\\ \n"
        latex_code += row
    
    latex_code += """\\hline
\\end{tabular}
\\end{table}"""
    
    return latex_code

# dataset = 'PubMed'
# num_features = 500
# num_classes = 3

# dataset = 'Cora'
# num_features = 1433
# num_classes = 7

dataset = 'Citeseer'
num_features = 3703
num_classes = 6


constants_file = os.path.join(os.path.dirname(__file__), 'constants.yaml')
with open(constants_file, 'r') as file:
    CONSTANTS = yaml.safe_load(file)
    
train_subgraph_sizes = CONSTANTS['train_sample_sizes'][dataset]
gnn_hidden_dims: list[int] = CONSTANTS['GNN_hidden_dims']
gnn_num_layers: list[int] = CONSTANTS['GNN_num_layers']
gnn_num_mlp_layers: list[int] = CONSTANTS['GNN_num_mlp_layers']
    
gnn_architectures: list[list[int]] = []
for (hidden_dim, num_layers, num_mlp_layers) in itertools.product(
    gnn_hidden_dims, gnn_num_layers, gnn_num_mlp_layers
):
    gnn_architectures.append([num_features, hidden_dim, num_classes, num_layers, num_mlp_layers])

input_file = 'experiments/Citeseer/20250506_110721/results_20250506_110721.pkl'

with open(input_file, 'rb') as f:
    data = pkl.load(f)

# print(data)
# breakpoint()

gnn_acc = data['gnn_acc'] * 100
gnn_transfer_acc = data['gnn_transfer_acc'] * 100
gnn_loss = data['gnn_loss']
gnn_transfer_loss = data['gnn_transfer_loss']
gntk_acc = data['gntk_acc'] * 100
gntk_transfer_acc = data['gntk_transfer_acc'] * 100
gntk_loss = data['gntk_loss']
gntk_transfer_loss = data['gntk_transfer_loss']

gnn_acc_mean = np.mean(gnn_acc, axis=0)
gnn_acc_std = np.std(gnn_acc, axis=0)
gnn_transfer_acc_mean = np.mean(gnn_transfer_acc, axis=0)
gnn_transfer_acc_std = np.std(gnn_transfer_acc, axis=0)
gnn_loss_mean = np.mean(gnn_loss, axis=0)
gnn_loss_std = np.std(gnn_loss, axis=0)
gnn_transfer_loss_mean = np.mean(gnn_transfer_loss, axis=0)
gnn_transfer_loss_std = np.std(gnn_transfer_loss, axis=0)
gntk_acc_mean = np.mean(gntk_acc, axis=0)
gntk_acc_std = np.std(gntk_acc, axis=0)
gntk_transfer_acc_mean = np.mean(gntk_transfer_acc, axis=0)
gntk_transfer_acc_std = np.std(gntk_transfer_acc, axis=0)
gntk_loss_mean = np.mean(gntk_loss, axis=0)
gntk_loss_std = np.std(gntk_loss, axis=0)
gntk_transfer_loss_mean = np.mean(gntk_transfer_loss, axis=0)
gntk_transfer_loss_std = np.std(gntk_transfer_loss, axis=0)


metrics = [
    ("GNN Test Accuracy", gnn_acc_mean.T, gnn_acc_std.T),
    ("GNN Transfer Accuracy", gnn_transfer_acc_mean.T, gnn_transfer_acc_std.T),
    # ("GNN Test Loss", gnn_loss_mean.T, gnn_loss_std.T),
    # ("GNN Transfer Loss", gnn_transfer_loss_mean.T, gnn_transfer_loss_std.T),
    ("GNTK Test Accuracy", gntk_acc_mean.T, gntk_acc_std.T),
    ("GNTK Transfer Accuracy", gntk_transfer_acc_mean.T, gntk_transfer_acc_std.T),
    # ("GNTK Test Loss", gntk_loss_mean.T, gntk_loss_std.T),
    # ("GNTK Transfer Loss", gntk_transfer_loss_mean.T, gntk_transfer_loss_std.T)
]

for name, mean, std in metrics:
    print(generate_latex_table(mean, std, name, range(1, len(gnn_architectures)+1), train_subgraph_sizes))
    print("\n")  # Add space between tables
