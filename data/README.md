Data

This project uses **League of Legends match telemetry data** in JSON format.  
To keep the repository lightweight and accessible:

-  A small **sample dataset** is included in the `data/` folder (`champion_matches.json`).  
-  The **full dataset** is not stored here (too large for GitHub).  

Using the full dataset

1. Download the full set of match JSON files (e.g. `matches1.json` … `matches10.json`) from your local copy or a provided link.  
2. Place all files in the `data/` folder so your project structure looks like:

   lol-role-synergy-analytics/
├─ data/
│ ├─ matches1.json
│ ├─ matches2.json
│ ├─ ...
│ ├─ matches10.json
│ └─ champion.json

3. Run the analysis:
```bash
python src/match-data-analysis.py
