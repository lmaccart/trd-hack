# 🏁 HackTheTrack Data Setup

## Quick Start (Local Development)

1. **Clone the repo**
```bash
   git clone https://github.com/your-team/hackthetrack.git
   cd hackthetrack
```

2. **Run setup (downloads data automatically)**
```bash
   chmod +x setup.sh
   ./setup.sh
```

3. **Start coding**
```python
   from race_data_loader import HackTheTrackDataLoader
   loader = HackTheTrackDataLoader()

   # Load data
   telemetry = loader.load('telemetry')
```

## Google Colab
```python
!pip install gdown
!gdown --folder https://drive.google.com/drive/folders/1IpIK0kjIoD3szP7qdXZU9978PyxFK81O -O data
```

## Data Location

- **Google Drive**: [HackTheTrack Data Folder](https://drive.google.com/drive/folders/1IpIK0kjIoD3szP7qdXZU9978PyxFK81O)
- **Local Cache**: `data/raw/` (after running setup)
- **Samples**: `data/samples/` (for fast development)

## Available Datasets

Run `loader.load()` to see all available datasets after download.

## Tips

- Use `sample_rows=1000` during development for faster iteration
- Data is cached locally - download only happens once
- Run `loader.info('dataset_name')` to explore any dataset

## Troubleshooting

If download fails:
1. Check internet connection
2. Try downloading manually from the Drive link
3. Place files in `data/raw/` directory
