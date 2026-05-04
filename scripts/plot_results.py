# import os
# import glob
# import pandas as pd
# import matplotlib.pyplot as plt

# def parse_log_for_table(filepath):
#     with open(filepath, 'r') as f:
#         lines = f.readlines()
    
#     table_lines = []
#     in_table = False
    
#     for line in lines:
#         line = line.strip()
#         # Find header line containing Sparsity and Accuracy
#         if line.startswith('|') and 'Sparsity' in line and 'Accuracy' in line:
#             in_table = True
#             table_lines.append(line)
#         elif in_table and line.startswith('|'):
#             table_lines.append(line)
#         elif in_table and not line.startswith('|'):
#             # Stop if we've collected data rows
#             if len(table_lines) > 2:
#                 break
#             else:
#                 table_lines = []
#                 in_table = False
            
#     if len(table_lines) <= 2:
#         return None
        
#     # Process the table
#     header = [x.strip() for x in table_lines[0].split('|')[1:-1]]
#     data = []
#     for line in table_lines[2:]:  # skip header and separator
#         row = [x.strip() for x in line.split('|')[1:-1]]
#         data.append(row)
        
#     df = pd.DataFrame(data, columns=header)
#     for col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
        
#     return df

# def plot_experiments(log_files, output_path):
#     plt.figure(figsize=(10, 6))
    
#     # Mapping log filenames to cleaner labels for the plot
#     labels = {
#         'threshold_sweep_final.log': 'W16A16 Threshold',
#         'threshold_sweep_final_v2.log': 'W16A16 Threshold v2',
#         'threshold_sweep_4bit_final.log': 'W4A4 Threshold',
#         'w4a4_sweep.log': 'W4A4 Sweep',
#         'sweep_output.log': 'QSVD Sweep',
#         'threshold_sweep.log': 'Initial Threshold Sweep'
#     }
    
#     has_data = False
#     for filepath in log_files:
#         df = parse_log_for_table(filepath)
#         if df is not None and not df.empty:
#             basename = os.path.basename(filepath)
#             label = labels.get(basename, basename)
            
#             # Ensure plotting columns exist
#             if 'Sparsity' in df.columns and 'Accuracy' in df.columns:
#                 df = df.sort_values(by='Sparsity')
#                 plt.plot(df['Sparsity'], df['Accuracy'], marker='o', label=label, linewidth=2)
#                 has_data = True
            
#     if not has_data:
#         print("No valid data tables found in the log files.")
#         return
        
#     plt.xlabel('Sparsity (%)', fontsize=14)
#     plt.ylabel('Accuracy', fontsize=14)
#     plt.title('Accuracy vs. Sparsity across Compression Strategies', fontsize=16)
#     plt.legend(fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     print(f"Plot saved successfully to {output_path}")

# if __name__ == "__main__":
#     base_dir = "/data1/vikram/QSVD/QSVD"
#     log_files = glob.glob(os.path.join(base_dir, "*.log"))
#     output_png = os.path.join(base_dir, "figs", "accuracy_vs_sparsity.png")
    
#     # Ensure figs directory exists
#     os.makedirs(os.path.dirname(output_png), exist_ok=True)
    
#     plot_experiments(log_files, output_png)


# import matplotlib.pyplot as plt

# threshold = [0, 10, 20, 30, 50]
# accuracy4 = [0.6143, 0.6227, 0.5949, 0.5483, 0.3718]
# accuracy8 = [0.6427, 0.6317, 0.6301, 0.5991, 0.4308]
# accuracy16 = [0.6905, 0.6897, 0.6832, 0.6656, 0.6371]

# plt.figure()
# plt.plot(threshold, accuracy4, marker='o', label='W4A4')
# plt.plot(threshold, accuracy8, marker='o', label='W8A8')
# plt.plot(threshold, accuracy16, marker='o', label='W16A16')
# plt.xlabel("Threshold (%)")
# plt.ylabel("Accuracy")
# plt.title("Threshold vs Accuracy")
# plt.legend()
# plt.grid()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

threshold = [0, 10, 20, 30, 50]
sparsity = [0.00, 9.97, 19.94, 29.91, 49.90]
accuracy = [0.6143, 0.6227, 0.5949, 0.5483, 0.3718]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(threshold, sparsity, accuracy)

ax.set_xlabel("Threshold (%)")
ax.set_ylabel("Sparsity (%)")
ax.set_zlabel("Accuracy")
ax.set_title("Threshold vs Sparsity vs Accuracy (w4A4)")

# plt.show()
output_path="/data1/vikram/QSVD/QSVD/figs/threshold_vs_accuracy4.png"
plt.savefig(output_path, dpi=300)
print(f"Plot saved successfully to {output_path}")

# plt.show()