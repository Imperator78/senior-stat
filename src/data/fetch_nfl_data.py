"""
fetch_nfl_data.py

This file contains functions to fetch and preprocess NFL game data using the nfl_data_py library.
It retrieves relevant statistics for the Washington Commanders and Chicago Bears, focusing on 4th down situations.
"""

import nfl_data_py as nfl
import pandas as pd

def fetch_nfl_data(team_names=['Washington Commanders', 'Chicago Bears']):
    """
    Fetch NFL game data for specified teams and filter for 4th down situations.
    
    Parameters:
    team_names (list): List of team names to fetch data for.
    
    Returns:
    pd.DataFrame: DataFrame containing relevant 4th down statistics.
    """
    # Fetch all game data
    games = nfl.import_pbp_data()

    # Filter for the specified teams and 4th down situations
    filtered_games = games[(games['team'].isin(team_names)) & (games['down'] == 4)]

    # Select relevant columns for analysis
    relevant_columns = [
        'game_id', 'season', 'week', 'team', 'yardline_100', 
        'yards_to_go', 'play_type', 'result', 'touchdown', 'field_goal'
    ]
    
    return filtered_games[relevant_columns]

def preprocess_data(df):
    """
    Preprocess the fetched data to calculate expected points for 4th down situations.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 4th down statistics.
    
    Returns:
    pd.DataFrame: Processed DataFrame with expected points calculations.
    """
    # Example preprocessing steps (to be customized based on analysis needs)
    df['expected_points'] = df.apply(calculate_expected_points, axis=1)
    
    return df

def calculate_expected_points(row):
    """
    Calculate expected points based on the result of the play.
    
    Parameters:
    row (pd.Series): A row from the DataFrame containing play information.
    
    Returns:
    float: Expected points for the play.
    """
    if row['result'] == 'touchdown':
        return 6.0  # Example value for touchdown
    elif row['result'] == 'field_goal':
        return 3.0  # Example value for field goal
    else:
        return -2.0  # Example value for turnover or failed attempt
"""