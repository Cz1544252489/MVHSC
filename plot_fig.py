import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

def analyze_and_plot_from_folder(folder_name):
    """
    Analyze the effect of orth_y with various factors (loop1, hypergrad_method, lr, mu) 
    on time_cost, last_UL_dval, and last_LL_dval, using all .txt files in the specified folder.

    Parameters:
    folder_name (str): The name of the folder containing .txt files (relative to the current working directory).
    """
    # Construct the folder path
    folder_path = os.path.join(os.getcwd(), folder_name)

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Find all .txt files in the specified folder
    txt_files = glob(os.path.join(folder_path, "*.txt"))
    if not txt_files:
        print(f"No .txt files found in the folder '{folder_name}'.")
        return

    for file_path in txt_files:
        print(f"Processing file: {os.path.basename(file_path)}")

        # Load the data from the file
        df = pd.read_csv(file_path, sep="\t")

        # Ensure correct data types for numerical columns
        for col in ['lr', 'mu', 'loop1', 'rloop0', 'time_cost', 'last_UL_dval', 'last_LL_dval']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Group data by orth_y with different combinations
        orth_y_loop1_analysis = df.groupby(['orth_y', 'loop1'])[['time_cost', 'last_UL_dval', 'last_LL_dval']].mean()
        orth_y_hypergrad_analysis = df.groupby(['orth_y', 'hypergrad_method'])[['time_cost', 'last_UL_dval', 'last_LL_dval']].mean()
        orth_y_lr_analysis = df.groupby(['orth_y', 'lr'])[['time_cost', 'last_UL_dval', 'last_LL_dval']].mean()
        orth_y_mu_analysis = df.groupby(['orth_y', 'mu'])[['time_cost', 'last_UL_dval', 'last_LL_dval']].mean()

        # Define line styles for orth_y
        line_styles = {True: 'solid', False: 'dashed'}
        colors = {'time_cost': 'blue', 'last_UL_dval': 'green', 'last_LL_dval': 'red'}

        # Plotting function
        def plot_and_save(group_analysis, x_label, file_suffix, title):
            fig, ax = plt.subplots(figsize=(12, 8))
            for orth_y, group_data in group_analysis.groupby(level=0):
                for metric, color in colors.items():
                    ax.plot(group_data.index.get_level_values(1), group_data[metric], 
                            label=f'{metric} (orth_y={orth_y})', linestyle=line_styles[orth_y], marker='o', color=color)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel("Metrics", fontsize=12)
            ax.legend()
            ax.grid(True)
            output_file = os.path.join(folder_path, f"{os.path.splitext(os.path.basename(file_path))[0]}_{file_suffix}.png")
            plt.savefig(output_file)
            plt.close()

        # Plot and save for all combinations
        plot_and_save(orth_y_loop1_analysis, "Loop1", "effect_of_orth_y_and_loop1_on_metrics", 
                      "Effect of orth_y and Loop1 on Metrics")
        plot_and_save(orth_y_hypergrad_analysis, "Hypergrad Method", "effect_of_orth_y_and_hypergrad_method_on_metrics", 
                      "Effect of orth_y and Hypergrad Method on Metrics")
        plot_and_save(orth_y_lr_analysis, "Learning Rate (lr)", "effect_of_orth_y_and_lr_on_metrics", 
                      "Effect of orth_y and Learning Rate on Metrics")
        plot_and_save(orth_y_mu_analysis, "Mu", "effect_of_orth_y_and_mu_on_metrics", 
                      "Effect of orth_y and Mu on Metrics")

        print(f"Plots for {os.path.basename(file_path)} have been saved to '{folder_path}'.")

# Example usage
analyze_and_plot_from_folder('gtest2')


