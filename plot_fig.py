import pandas as pd
import matplotlib.pyplot as plt

def analyze_and_plot(file_path):
    """
    Analyze the effect of orth_y with various factors (loop1, hypergrad_method, lr, mu) 
    on time_cost, last_UL_dval, and last_LL_dval, and save combined plots as PNG files.

    Parameters:
    file_path (str): Path to the input file in tab-separated format.
    """
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
    def plot_and_save(group_analysis, x_label, file_name, title):
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
        plt.savefig(file_name)
        plt.close()

    # Plot and save for all combinations
    plot_and_save(orth_y_loop1_analysis, "Loop1", "effect_of_orth_y_and_loop1_on_metrics.png", 
                  "Effect of orth_y and Loop1 on Metrics")
    plot_and_save(orth_y_hypergrad_analysis, "Hypergrad Method", "effect_of_orth_y_and_hypergrad_method_on_metrics.png", 
                  "Effect of orth_y and Hypergrad Method on Metrics")
    plot_and_save(orth_y_lr_analysis, "Learning Rate (lr)", "effect_of_orth_y_and_lr_on_metrics.png", 
                  "Effect of orth_y and Learning Rate on Metrics")
    plot_and_save(orth_y_mu_analysis, "Mu", "effect_of_orth_y_and_mu_on_metrics.png", 
                  "Effect of orth_y and Mu on Metrics")

    print("All plots with consistent colors for metrics have been saved as PNG files.")

# Example usage
# analyze_and_plot('data.txt')


# Example usage
analyze_and_plot('BDA2025-01-07_18-00-58.txt')

