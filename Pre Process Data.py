import pandas as pd
from pathlib import Path

files = list(Path('./Data/Original').glob('*.CSV'))
print(files)

#------------------------------------------------
# Add Sequence depend on time gap 
current_max = 0  # Global tracker for GroupID

output_dir = Path("./Data/Sequence") 
#Create the output folder if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)


for file_path in files:
    df = pd.read_csv(file_path)
    df['fDateTime'] = pd.to_datetime(df['fDateTime'])
    
    # Calculate new Sequence (starts from 0)
    new_sequences = (df['fDateTime'].diff() > pd.Timedelta(minutes=15)).cumsum()

    # Shift the IDs by the current_max + 1
    df['SequenceID'] = new_sequences + current_max + 1
    
    # Update the max for the next file
    current_max = df['SequenceID'].max()
    
    # Save the file (using previous logic)
    output_path = output_dir / f"{file_path.stem}_sequence.csv"
    df.to_csv(output_path, index=False)

print("\nAll files have been processed and saved.")

#----------------------------------------------------
# Define source (Sequence) and destination (Filtered) folders
input_dir = Path("./Data/Sequence")
output_dir = Path("./Data/Filtered")
output_dir.mkdir(parents=True, exist_ok=True)

# Define sensors that should not be zero
sensor_cols = ['InletT_PV', 'OutletT_PV','FlueGasT_PV', 'Flow_PV','RealPower']

for file_path in input_dir.glob('*.csv'):
    # Load the resampled data
    df = pd.read_csv(file_path)
    
    # FILTER 1: Remove rows where InletT_PV is below 150 (Outliers)
    # Only keep rows where InletT_PV >= 150
    df = df[df['InletT_PV'] >= 150]
    
    # FILTER 2: Remove rows where all main sensors are 0
    # This keeps a row if ANY of the sensor values are greater than zero
    df = df[(df[sensor_cols] > 0).any(axis=1)]
    
    # Save to the 'Filtered' folder
    if not df.empty:
        output_path = output_dir / f"{file_path.stem}_filtered.csv"
        df.to_csv(output_path, index=False)
        print(f"Processed {file_path.name}: {len(df)} valid rows remaining.")
    else:
        print(f"Skipped {file_path.name}: No data remained after filtering.")

print("\nFiltering complete. Clean data is in '../Data/Filtered'.")


#-------------------------------------------------------
# Define source and destination directories
input_dir = Path('./Data/Filtered')
output_dir = Path("./Data/Resample")             

# Create the output folder if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Loop through all CSV files in the input directory
# Use *.csv to find all csv files
for file_path in input_dir.glob('*.csv'):
    print(f"Processing: {file_path.name}")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert to datetime and set as index
    df['fDateTime'] = pd.to_datetime(df['fDateTime'])
    df = df.set_index('fDateTime')
    
    # Aggregate by minute, calculate mean, and round to 1 decimal
    df_aggregated = df.resample('1Min').mean(numeric_only=True).round(1)
    
    # Clean up: remove empty minutes and reset index
    df_aggregated = df_aggregated.dropna().reset_index()
    
    # Create the output path
    # This keeps the original filename and adds '_resample'
    output_path = output_dir / f"{file_path.stem}_resample.csv"
    
    # Save the result
    df_aggregated.to_csv(output_path, index=False)

print("\nAll files have been processed and saved.")


#--------------------------------------------------------------
# Consolidate multiple file if any
# Define folder paths
input_dir = Path("./Data/Resample")
output_dir = Path("./Data/Consolidated")

# Create the folder if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "Master_Training_Data.csv"

# Collect all filtered CSV files
all_data = []
for file_path in input_dir.glob('*.csv'):
    df = pd.read_csv(file_path)
    all_data.append(df)
    print(f"Adding: {file_path.name}")

if all_data:
    # Merge all data into one master dataframe
    master_df = pd.concat(all_data, ignore_index=True)

    # Sort by time to ensure chronological order for AI
    master_df['fDateTime'] = pd.to_datetime(master_df['fDateTime'])
    master_df = master_df.sort_values(by='fDateTime')


    # DROP the SequenceID column
    # axis=1 means column, errors='ignore' prevents crashing if column is already missing
    master_df = master_df.drop(columns=['SequenceID'], errors='ignore')
    
    # Save to the new folder
    master_df.to_csv(output_file, index=False)

    print(f"\n--- Process Complete ---")
    print(f"Total files merged: {len(all_data)}")
    print(f"Total rows in master file: {len(master_df)}")
    print(f"Saved to: {output_file}")
else:
    print("No CSV files found in the Filtered folder.")


#-----------------------------------------------------------
# Add final sequence no after consolidate

files = list(Path('./Data/Consolidated').glob('*.CSV'))

output_dir = Path("./Data/Final")

# Create the folder if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)


current_max = 0  # Global tracker for GroupID

for file_path in files:
    df = pd.read_csv(file_path)
    df['fDateTime'] = pd.to_datetime(df['fDateTime'])
    
    # Calculate new Sequence (starts from 0)
    new_sequences = (df['fDateTime'].diff() > pd.Timedelta(minutes=15)).cumsum()
    
    # Shift the IDs by the current_max + 1
    df['SequenceID'] = new_sequences + current_max + 1
    
    # Update the max for the next file
    current_max = df['SequenceID'].max()
    
    # Save the file (using previous logic)
    output_path = output_dir / f"{file_path.stem}_Sequence.csv"
    df.to_csv(output_path, index=False)

#----------------------------------------------------------
# Remove shord data groups
# Define paths
input_file = Path("./Data/Final/Master_Training_Data_Sequence.csv")
output_dir = Path("./Data/Final")
output_file = output_dir / "Master_Training_Data_Final.csv"

# Load the consolidated sequence data
df = pd.read_csv(input_file)

# Size Filtering: Keep only sequences with at least 60 minutes of data
# This removes your short groups (like IDs 11-16) that lack enough context for LSTM
min_rows = 60
df_filtered = df.groupby('SequenceID').filter(lambda x: len(x) >= min_rows)

# Drop SequenceID column before saving for AI training
# sort one last time by the ID and Date to ensure order before dropping
df_filtered['fDateTime'] = pd.to_datetime(df_filtered['fDateTime'])
df_filtered = df_filtered.sort_values(by=['SequenceID', 'fDateTime'])
#df_final = df_filtered.drop(columns=['SequenceID'])

# Save the final cleaned dataset
#df_final.to_csv(output_file, index=False)
df_filtered.to_csv(output_file, index=False)

print(f"Original Row Count: {len(df)}")
print(f"Filtered Row Count: {len(df_filtered)}")
print(f"Removed {len(df) - len(df_filtered)} rows from small sequences.")
print(f"Final file saved to: {output_file}")
