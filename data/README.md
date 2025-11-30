# Data Directory

## Structure

- `raw/` - Original, unprocessed data from Oracle's Elixir
- `processed/` - Cleaned and transformed datasets

## Data Pipeline

1. **Raw Data** (`raw/2024_LoL_esports_match_data_from_OraclesElixir.csv`)
   - Source: https://oracleselixir.com/tools/downloads
   - Contains both player-level and team-level statistics
   - ~186,000 rows total

2. **Team Data** (`processed/team_data_2024.csv`)
   - Filtered to team-level observations only
   - Removes individual player stats

3. **Complete Team Data** (`processed/complete_team_data_2024.csv`)
   - Only includes complete observations (no missing data)

4. **Team Metrics** (`processed/team_metrics_data_2024.csv`)
   - Selected 60+ relevant features for modeling
   - Includes objectives, economy, vision, teamfight metrics

5. **PCA Transformed** (`processed/pca_transformed_data.csv`)
   - 9 principal components from dimensionality reduction
   - Ready for model training

## Downloading Data

Download the latest data from Oracle's Elixir:
```bash
# Visit https://oracleselixir.com/tools/downloads
# Download "2024_LoL_esports_match_data_from_OraclesElixir.csv"
# Place in data/raw/
```

## Regenerating Processed Data

```bash
python scripts/run_preprocessing.py
python scripts/run_feature_engineering.py
```
