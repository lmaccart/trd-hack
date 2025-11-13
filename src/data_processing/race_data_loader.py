import gdown
import pandas as pd
from pathlib import Path
import json
from typing import Optional, Dict, Union

class HackTheTrackDataLoader:
    """Data loader for HackTheTrack hackathon datasets"""

    # Your shared folder
    FOLDER_URL = 'https://drive.google.com/drive/folders/1IpIK0kjIoD3szP7qdXZU9978PyxFK81O?usp=sharing'

    def __init__(self, data_dir: str = 'data/raw', use_cache: bool = True):
        """
        Initialize the HackTheTrack data loader

        Args:
            data_dir: Directory to store downloaded files
            use_cache: Whether to use cached files if they exist
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.manifest_file = self.data_dir / '.manifest.json'
        self.datasets = {}

    def download_all(self, force: bool = False) -> Dict[str, Path]:
        """
        Download all files from the HackTheTrack Google Drive folder

        Args:
            force: Force re-download even if files exist

        Returns:
            Dictionary mapping dataset names to file paths
        """
        # Check for existing files
        if self.use_cache and not force:
            existing_files = list(self.data_dir.glob('*.csv')) + \
                           list(self.data_dir.glob('*.xlsx')) + \
                           list(self.data_dir.glob('*.json'))

            if existing_files:
                print(f"✓ Found {len(existing_files)} cached files")
                print("  Use force=True to re-download")
                return self._index_local_files()

        print("📥 Downloading HackTheTrack datasets from Google Drive...")
        print(f"   URL: {self.FOLDER_URL}")

        try:
            # Download the entire folder
            gdown.download_folder(
                url=self.FOLDER_URL,
                output=str(self.data_dir),
                quiet=False,
                use_cookies=False
            )

            print("✅ Download complete!")

        except Exception as e:
            print(f"⚠️ Download error: {e}")
            print("Trying alternative method...")
            # Alternative: manual file IDs if folder download fails
            return self._download_individual_files()

        return self._index_local_files()

    def _download_individual_files(self) -> Dict[str, Path]:
        """Fallback method: download files individually if folder download fails"""
        # If gdown folder download fails, you can add individual file IDs here
        # Extract these from the share links of individual files
        individual_files = {
            # 'filename': 'file_id',
            # Add as needed based on what's in your folder
        }

        print("ℹ️ Attempting individual file downloads...")
        for filename, file_id in individual_files.items():
            url = f'https://drive.google.com/uc?id={file_id}'
            output_path = self.data_dir / filename
            if not output_path.exists() or not self.use_cache:
                print(f"  Downloading {filename}...")
                gdown.download(url, str(output_path), quiet=False)

        return self._index_local_files()

    def _index_local_files(self) -> Dict[str, Path]:
        """Index all data files in the local directory"""
        self.datasets = {}

        # Look for all data files
        patterns = ['*.csv', '*.xlsx', '*.json', '*.parquet']
        all_files = []
        for pattern in patterns:
            all_files.extend(self.data_dir.glob(pattern))

        if not all_files:
            print("⚠️ No data files found. Run download_all() first.")
            return self.datasets

        print("\n📊 Available datasets:")
        manifest = {}

        for filepath in sorted(all_files):
            name = filepath.stem  # filename without extension
            self.datasets[name] = filepath

            # Get file info
            size_bytes = filepath.stat().st_size
            size_mb = size_bytes / (1024 * 1024)

            manifest[name] = {
                'path': str(filepath.relative_to(Path.cwd())),
                'size_mb': round(size_mb, 2),
                'type': filepath.suffix,
                'rows': None  # Will be filled when loaded
            }

            print(f"  ✓ {name:<30} ({size_mb:>8.2f} MB) [{filepath.suffix}]")

        # Save manifest
        with open(self.manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        return self.datasets

    def load(self,
             dataset_name: Optional[str] = None,
             sample_rows: Optional[int] = None,
             **kwargs) -> Optional[pd.DataFrame]:
        """
        Load a specific dataset as a pandas DataFrame

        Args:
            dataset_name: Name of the dataset (without extension)
                         If None, lists available datasets
            sample_rows: Number of rows to load (for development)
            **kwargs: Additional arguments passed to pd.read_csv/read_excel

        Returns:
            DataFrame or None if listing datasets
        """
        # Ensure we have indexed files
        if not self.datasets:
            self.download_all()

        # List datasets if no name provided
        if dataset_name is None:
            print("\n📚 Available datasets:")
            print("-" * 50)
            for name, path in self.datasets.items():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  • {name:<30} ({size_mb:.1f} MB)")
            print("-" * 50)
            print("\nUsage: df = loader.load('dataset_name')")
            return None

        # Check if dataset exists
        if dataset_name not in self.datasets:
            print(f"❌ Dataset '{dataset_name}' not found!")
            print(f"Available: {list(self.datasets.keys())}")
            return None

        filepath = self.datasets[dataset_name]

        # Load based on file type
        print(f"Loading {dataset_name}...", end=" ")

        try:
            if filepath.suffix == '.csv':
                if sample_rows:
                    df = pd.read_csv(filepath, nrows=sample_rows, **kwargs)
                else:
                    df = pd.read_csv(filepath, **kwargs)

            elif filepath.suffix in ['.xlsx', '.xls']:
                if sample_rows:
                    df = pd.read_excel(filepath, nrows=sample_rows, **kwargs)
                else:
                    df = pd.read_excel(filepath, **kwargs)

            elif filepath.suffix == '.json':
                df = pd.read_json(filepath, **kwargs)

            elif filepath.suffix == '.parquet':
                df = pd.read_parquet(filepath, **kwargs)
                if sample_rows:
                    df = df.head(sample_rows)

            else:
                print(f"❌ Unsupported file type: {filepath.suffix}")
                return None

            print(f"✓ ({len(df):,} rows, {len(df.columns)} columns)")
            return df

        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")
            return None

    def load_all(self, sample_rows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load all datasets into a dictionary"""
        if not self.datasets:
            self.download_all()

        all_data = {}
        for name in self.datasets.keys():
            df = self.load(name, sample_rows=sample_rows)
            if df is not None:
                all_data[name] = df

        return all_data

    def create_samples(self, sample_size: int = 1000):
        """Create sample files for faster development"""
        samples_dir = Path('data/samples')
        samples_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🔬 Creating sample datasets ({sample_size} rows each)...")

        for name, filepath in self.datasets.items():
            df = self.load(name, sample_rows=sample_size)
            if df is not None:
                sample_filename = f'sample_{filepath.name}'
                sample_path = samples_dir / sample_filename

                if filepath.suffix == '.csv':
                    df.to_csv(sample_path, index=False)
                elif filepath.suffix in ['.xlsx', '.xls']:
                    df.to_excel(sample_path, index=False)

                print(f"  ✓ Created {sample_path}")

    def info(self, dataset_name: str):
        """Display detailed information about a dataset"""
        df = self.load(dataset_name)
        if df is None:
            return

        print(f"\n📊 Dataset: {dataset_name}")
        print("=" * 60)
        print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"\nColumns:")
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            null_pct = (1 - non_null/len(df)) * 100
            print(f"  • {col:<30} {dtype:<15} ({null_pct:.1f}% null)")

        print(f"\nFirst 5 rows:")
        print(df.head())
