# Cognitive Decline Pattern Analysis in Alzheimer's Disease

## Project Overview
This project analyzes cognitive decline patterns in Alzheimer's Disease using machine learning techniques on the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset. We identify and characterize five distinct progression patterns to better understand disease trajectories and potential intervention points.

## Key Features
- Multi-dimensional analysis of cognitive decline patterns
- Five distinct progression profiles identification
- Comprehensive visualization of cognitive trajectories
- Statistical validation of cluster patterns
- Integration of multiple cognitive assessment metrics

## Project Structure
```
├── data_processing/
│   ├── preprocess.py         # Data preprocessing and cleaning
│   └── feature_engineering.py # Feature creation and scaling
├── models/
│   ├── clustering.py         # Clustering algorithms implementation
│   └── validation.py         # Model validation and evaluation
├── visualization/
│   ├── plots.py             # Visualization functions
│   └── analysis.py          # Results analysis and interpretation
├── utils/
│   └── helpers.py           # Utility functions
├── main.py                  # Main execution script
├── config.py               # Configuration parameters
└── requirements.txt        # Project dependencies
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Data Preprocessing:
```bash
python data_processing/preprocess.py
```

2. Run Analysis:
```bash
python main.py
```

## Results
The analysis identifies five distinct cognitive decline profiles:
1. Stable Cognition (SC)
2. Mild Progressive Decline (MPD)
3. Moderate Progressive Decline (MoPD)
4. Rapid Progressive Decline (RPD)
5. Severe Progressive Decline (SPD)

## Dependencies
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- SciPy

## References
1. ADNI Database
2. Related research papers and methodologies
3. Statistical analysis techniques

## License
MIT License

## Contributors
- Vidyadhar Dinde
- Manali Khedekar
- Adiya Khandare
- Kranti Varekar
- Anuja Suntnur
- Sneha Kumbhar

## Acknowledgments
- ADNI for providing the dataset
- Research community for methodological insights