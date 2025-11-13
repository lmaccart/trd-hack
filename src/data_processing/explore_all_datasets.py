"""
Explore all datasets to verify format consistency and identify differences
"""
import pandas as pd
import os
from pathlib import Path
import json

print("="*80)
print("EXPLORING ALL RACE DATASETS")
print("="*80)

# Define all tracks
tracks = {
    'barber': 'Barber Motorsports Park',
    'COTA': 'Circuit of the Americas',
    'indianapolis': 'Indianapolis Motor Speedway',
    'road-america': 'Road America',
    'sebring': 'Sebring International Raceway',
    'Sonoma': 'Sonoma Raceway',
    'virginia-international-raceway': 'Virginia International Raceway'
}

base_path = Path('../../data/raw/datasets')
results = {}

for track_key, track_name in tracks.items():
    print(f"\n{'='*80}")
    print(f"TRACK: {track_name} ({track_key})")
    print(f"{'='*80}")

    track_path = base_path / track_key

    if not track_path.exists():
        print(f"⚠️  Track directory not found: {track_path}")
        continue

    # Find all CSV files for this track
    csv_files = list(track_path.rglob('*.csv'))

    print(f"\nFound {len(csv_files)} CSV files")

    # Categorize files
    telemetry_files = []
    lap_time_files = []
    lap_start_files = []
    lap_end_files = []
    other_files = []

    for f in csv_files:
        fname = f.name.lower()
        if 'telemetry' in fname:
            telemetry_files.append(f)
        elif 'lap_time' in fname:
            lap_time_files.append(f)
        elif 'lap_start' in fname:
            lap_start_files.append(f)
        elif 'lap_end' in fname:
            lap_end_files.append(f)
        else:
            other_files.append(f)

    print(f"  - Telemetry files: {len(telemetry_files)}")
    print(f"  - Lap time files: {len(lap_time_files)}")
    print(f"  - Lap start files: {len(lap_start_files)}")
    print(f"  - Lap end files: {len(lap_end_files)}")
    print(f"  - Other files: {len(other_files)}")

    # Analyze telemetry files
    track_results = {
        'track_name': track_name,
        'races': {}
    }

    for race_num in [1, 2]:
        race_key = f'Race {race_num}'
        race_telemetry = [f for f in telemetry_files if f'R{race_num}' in f.name or f'Race {race_num}' in str(f.parent)]
        race_lap_time = [f for f in lap_time_files if f'R{race_num}' in f.name or f'Race {race_num}' in str(f.parent)]

        if not race_telemetry:
            continue

        print(f"\n  Race {race_num}:")

        # Load and analyze telemetry
        telem_file = race_telemetry[0]
        print(f"    Telemetry: {telem_file.name}")

        try:
            df_telem = pd.read_csv(telem_file, nrows=1000)  # Sample for speed

            race_data = {
                'telemetry_file': str(telem_file.relative_to(base_path)),
                'telemetry_columns': list(df_telem.columns),
                'telemetry_sample_rows': len(df_telem),
            }

            print(f"      Columns: {', '.join(df_telem.columns[:10])}{'...' if len(df_telem.columns) > 10 else ''}")
            print(f"      Total columns: {len(df_telem.columns)}")

            # Check if it's long or wide format
            if 'telemetry_name' in df_telem.columns and 'telemetry_value' in df_telem.columns:
                print(f"      Format: LONG (pivot required)")
                unique_telemetry = df_telem['telemetry_name'].unique()
                print(f"      Unique telemetry types: {len(unique_telemetry)}")
                race_data['format'] = 'long'
                race_data['telemetry_types'] = list(unique_telemetry)
            else:
                print(f"      Format: WIDE (already pivoted)")
                race_data['format'] = 'wide'

            # Load lap time if available
            if race_lap_time:
                lap_file = race_lap_time[0]
                print(f"    Lap times: {lap_file.name}")
                df_lap = pd.read_csv(lap_file, nrows=100)

                race_data['lap_time_file'] = str(lap_file.relative_to(base_path))
                race_data['lap_time_columns'] = list(df_lap.columns)

                print(f"      Columns: {', '.join(df_lap.columns)}")

            track_results['races'][race_key] = race_data

        except Exception as e:
            print(f"    ⚠️  Error loading: {e}")

    results[track_key] = track_results

# Summary
print(f"\n\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

print(f"\nTracks analyzed: {len(results)}")

for track_key, data in results.items():
    print(f"\n{data['track_name']}:")
    print(f"  Races: {len(data['races'])}")
    for race_key, race_data in data['races'].items():
        print(f"    {race_key}: Format={race_data.get('format', 'unknown')}")

# Save results
output_file = '../../data/results/dataset_exploration.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n\nDetailed results saved to: {output_file}")
print("="*80)
