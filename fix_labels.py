import pandas as pd

def fix_season_labels(input_file, output_file):
    # Read the dataset
    df = pd.read_csv(input_file)

    # Define a mapping for Season labels
    season_mapping = {
        'Kharif     ': 0,
        'Rabi       ': 1,
        'Whole Year ': 2,
        'Summer     ': 3,
        'Winter     ': 4,
        'Autumn     ': 5,
        'Zaid       ': 6
    }

    # Apply the mapping to fix the labels
    df['Season'] = df['Season'].map(season_mapping)

    # Save the fixed dataset to a new CSV file
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Define input and output file paths
    input_file = 'data/crop_yield_data.csv'
    output_file = 'data/crop_yield_data_fixed.csv'

    # Call the function to fix labels
    fix_season_labels(input_file, output_file)
