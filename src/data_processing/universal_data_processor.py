"""
Universal data processor for all race tracks
Processes all tracks and races into a unified format
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple

class UniversalRaceDataProcessor:
    """Process race data from all tracks into a unified format"""

    TRACKS = {
        'barber': 'Barber Motorsports Park',
        'COTA': 'Circuit of the Americas',
        'indianapolis': 'Indianapolis Motor Speedway',
        'road-america': 'Road America',
        'sebring': 'Sebring International Raceway',
        'Sonoma': 'Sonoma Raceway',
        'virginia-international-raceway': 'Virginia International Raceway (VIR)'
    }

    def __init__(self, base_path='../../data/raw/datasets'):
        self.base_path = Path(base_path)
        self.results = {}

    def find_files(self, track_key: str, race_num: int) -> Dict[str, Path]:
        """Find telemetry and lap time files for a specific track and race"""
        track_path = self.base_path / track_key

        # Find files matching race number
        telemetry_files = list(track_path.rglob(f'*telemetry*.csv'))
        lap_time_files = list(track_path.rglob(f'*lap_time*.csv'))

        # Filter by race number
        race_telemetry = [f for f in telemetry_files if f'R{race_num}' in f.name or f'Race {race_num}' in str(f.parent)]
        race_lap_time = [f for f in lap_time_files if f'R{race_num}' in f.name or f'Race {race_num}' in str(f.parent)]

        return {
            'telemetry': race_telemetry[0] if race_telemetry else None,
            'lap_time': race_lap_time[0] if race_lap_time else None
        }

    def process_telemetry(self, file_path: Path) -> pd.DataFrame:
        """Load and process telemetry data, handling both long and wide formats"""
        print(f"    Loading telemetry: {file_path.name}")
        df = pd.read_csv(file_path)

        # Check if it's in long format (needs pivoting)
        if 'telemetry_name' in df.columns and 'telemetry_value' in df.columns:
            print(f"      Converting from LONG to WIDE format...")

            # Pivot the data
            df_wide = df.pivot_table(
                index=['timestamp', 'lap', 'vehicle_id', 'outing'],
                columns='telemetry_name',
                values='telemetry_value',
                aggfunc='first'
            ).reset_index()
            df_wide.columns.name = None

            # Add vehicle_number if it exists in original
            if 'vehicle_number' in df.columns:
                vehicle_map = df[['vehicle_id', 'vehicle_number']].drop_duplicates()
                df_wide = df_wide.merge(vehicle_map, on='vehicle_id', how='left')

            print(f"      Pivoted to {len(df_wide)} rows x {len(df_wide.columns)} columns")
            return df_wide
        else:
            print(f"      Already in WIDE format")
            return df

    def process_lap_times(self, file_path: Path) -> pd.DataFrame:
        """Load and process lap time data"""
        print(f"    Loading lap times: {file_path.name}")
        df = pd.read_csv(file_path)

        # Standardize column names
        if 'value' in df.columns and 'timestamp' not in df.columns:
            df = df.rename(columns={'value': 'timestamp'})

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['vehicle_id', 'timestamp'])

        # Calculate lap times (difference between consecutive lap end timestamps)
        df['lap_time_seconds'] = df.groupby('vehicle_id')['timestamp'].diff().dt.total_seconds()

        # Remove invalid laps (lap 32768 is warmup)
        df_clean = df[df['lap'] < 32768].copy()

        return df_clean

    def merge_telemetry_and_laptimes(self, df_telemetry: pd.DataFrame, df_laptimes: pd.DataFrame) -> pd.DataFrame:
        """Merge telemetry and lap time data"""
        print(f"    Merging telemetry and lap times...")

        # Aggregate telemetry by lap
        telemetry_cols = [col for col in df_telemetry.columns
                         if col not in ['timestamp', 'lap', 'vehicle_id', 'vehicle_number', 'outing']]

        agg_dict = {col: 'mean' for col in telemetry_cols if df_telemetry[col].dtype in ['float64', 'int64']}

        if not agg_dict:
            print("      ⚠️  No numeric telemetry columns found")
            return None

        # Group by vehicle and lap
        group_cols = ['vehicle_id', 'lap']
        if 'vehicle_number' in df_telemetry.columns:
            group_cols.append('vehicle_number')

        telemetry_agg = df_telemetry.groupby(group_cols).agg(agg_dict).reset_index()

        # Merge with lap times
        merge_cols = ['vehicle_id', 'lap']
        if 'vehicle_number' in df_laptimes.columns and 'vehicle_number' in telemetry_agg.columns:
            merge_cols = ['vehicle_id', 'vehicle_number', 'lap']

        merged = df_laptimes.merge(telemetry_agg, on=merge_cols, how='inner')

        print(f"      Merged: {len(merged)} laps")
        return merged

    def process_track_race(self, track_key: str, track_name: str, race_num: int) -> Tuple[pd.DataFrame, Dict]:
        """Process a single track and race"""
        print(f"\n  Race {race_num}:")

        files = self.find_files(track_key, race_num)

        if not files['telemetry'] or not files['lap_time']:
            print(f"    ⚠️  Missing files")
            return None, None

        try:
            # Process telemetry
            df_telemetry = self.process_telemetry(files['telemetry'])

            # Process lap times
            df_laptimes = self.process_lap_times(files['lap_time'])

            # Merge
            df_merged = self.merge_telemetry_and_laptimes(df_telemetry, df_laptimes)

            if df_merged is None:
                return None, None

            # Add metadata
            df_merged['track'] = track_key
            df_merged['track_name'] = track_name
            df_merged['race'] = race_num

            # Calculate stats
            stats = {
                'total_laps': len(df_merged),
                'vehicles': int(df_merged['vehicle_id'].nunique()),
                'avg_lap_time': float(df_merged['lap_time_seconds'].mean()) if 'lap_time_seconds' in df_merged.columns else None,
                'min_lap_time': float(df_merged['lap_time_seconds'].min()) if 'lap_time_seconds' in df_merged.columns else None,
                'max_lap_time': float(df_merged['lap_time_seconds'].max()) if 'lap_time_seconds' in df_merged.columns else None,
                'columns': list(df_merged.columns)
            }

            print(f"      ✓ {stats['total_laps']} laps, {stats['vehicles']} vehicles")
            print(f"      ✓ Lap time: {stats['min_lap_time']:.2f}s - {stats['max_lap_time']:.2f}s (avg: {stats['avg_lap_time']:.2f}s)")

            return df_merged, stats

        except Exception as e:
            print(f"    ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def process_all(self) -> Dict:
        """Process all tracks and races"""
        print("="*80)
        print("UNIVERSAL DATA PROCESSING - ALL TRACKS")
        print("="*80)

        all_data = []
        summary = {}

        for track_key, track_name in self.TRACKS.items():
            print(f"\n{'='*80}")
            print(f"TRACK: {track_name}")
            print(f"{'='*80}")

            track_summary = {'races': {}}

            for race_num in [1, 2]:
                df, stats = self.process_track_race(track_key, track_name, race_num)

                if df is not None:
                    all_data.append(df)
                    track_summary['races'][f'Race {race_num}'] = stats

            summary[track_key] = track_summary

        # Combine all data
        if all_data:
            print(f"\n{'='*80}")
            print("COMBINING ALL DATA")
            print(f"{'='*80}")

            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n✓ Combined dataset: {len(combined_df)} total laps")
            print(f"✓ Tracks: {combined_df['track'].nunique()}")
            print(f"✓ Vehicles: {combined_df['vehicle_id'].nunique()}")
            print(f"✓ Columns: {len(combined_df.columns)}")

            # Save combined data
            output_file = '../../data/processed/all_tracks_combined.csv'
            combined_df.to_csv(output_file, index=False)
            print(f"\n✓ Saved: {output_file}")

            # Save summary
            summary_file = '../../data/results/all_tracks_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✓ Saved: {summary_file}")

            return combined_df, summary
        else:
            print("\n❌ No data processed")
            return None, None

if __name__ == '__main__':
    processor = UniversalRaceDataProcessor()
    df, summary = processor.process_all()

    if df is not None:
        print(f"\n{'='*80}")
        print("SUCCESS!")
        print(f"{'='*80}")
        print(f"\nProcessed {len(df)} laps across {df['track'].nunique()} tracks")
        print(f"Ready for analysis!")
