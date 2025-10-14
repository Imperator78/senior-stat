import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data.fetch_nfl_data import fetch_nfl_data
from src.analysis.ep_heatmap import calculate_expected_points

def plot_heatmap():
    # Fetch NFL data for the Washington Commanders and Chicago Bears
    data = fetch_nfl_data(['Washington Commanders', 'Chicago Bears'])
    
    # Calculate expected points for going for it on 4th down
    ep_data = calculate_expected_points(data)
    
    # Create a pivot table for the heatmap
    heatmap_data = ep_data.pivot("yards_to_goal", "yards_to_first_down", "expected_points")
    
    # Set up the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Expected Points'})
    
    # Set labels and title
    plt.title('Expected Points Heatmap for 4th Down Decisions\nWashington Commanders vs Chicago Bears', fontsize=16)
    plt.xlabel('Yards to First Down', fontsize=12)
    plt.ylabel('Yards to Goal', fontsize=12)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_heatmap()