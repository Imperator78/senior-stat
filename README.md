# NFL 4th Down Heatmap

This project analyzes 4th down decision-making for the Washington Commanders and the Chicago Bears using historical NFL game data. It calculates the expected points (EP) for going for it on 4th down and visualizes the results in a heatmap format.

## Project Structure

```
nfl-4th-down-heatmap
├── src
│   ├── data
│   │   └── fetch_nfl_data.py       # Functions to fetch and preprocess NFL game data
│   ├── analysis
│   │   └── ep_heatmap.py            # Functions to calculate expected points for 4th down
│   └── viz
│       └── plot_heatmap.py          # Visualization of expected points as a heatmap
├── requirements.txt                  # Required Python packages
└── README.md                         # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd nfl-4th-down-heatmap
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Fetch the NFL data by running:
   ```
   python src/data/fetch_nfl_data.py
   ```

2. Calculate expected points for 4th down situations:
   ```
   python src/analysis/ep_heatmap.py
   ```

3. Generate the heatmap visualization:
   ```
   python src/viz/plot_heatmap.py
   ```

## Heatmap Visualization

The heatmap will display the expected points for going for it on 4th down, with:
- **X-axis**: Number of yards to the first down
- **Y-axis**: Number of yards to goal

This visualization helps in understanding the decision-making process for 4th down plays based on historical performance data.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License.