# League of Legends Esports Match Outcome Prediction

## Overview

This project explores professional League of Legends esports data to build machine learning models that predict match outcomes. Using 2024 competitive match data from Oracle's Elixir, I implemented and compared multiple classification models to understand which in-game factors most strongly correlate with victory.

The work started in Jupyter Notebooks for exploration and prototyping, then evolved into a modular Python package with proper software engineering practices including experiment tracking, automated pipelines, and comprehensive model evaluation.

## Project Motivation

League of Legends is a complex game where teams must balance multiple objectives: securing gold leads, controlling map objectives like dragons and barons, maintaining vision, and winning teamfights. 

This project investigates which factors are most predictive of match outcomes using machine learning, and explores how different modeling approaches handle the nonlinear relationships in competitive esports data.

## Repository Structure

```
├── data/
│   ├── raw/                    # Original CSV from Oracle's Elixir
│   └── processed/              # Pipeline-generated datasets (team_data, pca_transformed, etc.)
├── notebooks/                  # Jupyter notebooks for initial exploration
│   ├── EDA_and_Dataset_Processing.ipynb
│   ├── Naive_Bayes.ipynb
│   ├── Logistic_Regression.ipynb
│   ├── GBDT.ipynb
│   └── Tree_Models-BaggedDT_and_RF.ipynb
├── src/                        # Source code package
│   ├── config.py               # Configuration and constants
│   ├── data/                   # Data loading and preprocessing
│   ├── features/               # Feature engineering (PCA)
│   ├── models/                 # Model training pipeline
│   └── evaluation/             # Metrics and visualizations
├── scripts/                    # Executable pipeline scripts
├── models/                     # Saved model artifacts
├── results/                    # Evaluation plots and reports
├── requirements.txt
└── setup.py
```

## Key Features

**Data Processing Pipeline:**
- Automated pipeline filters 180K+ rows to team-level aggregated statistics
- Selected 60+ relevant features including objectives (dragons, barons, towers), economy (gold differential), vision control, and teamfight metrics
- Automated data quality checks and missing value handling

**Feature Engineering:**
- Applied PCA with permutation testing to validate statistical significance
- Reduced dimensionality to 9 interpretable components explaining major variance in match outcomes
- Components capture: gold advantage, vision control, combat effectiveness, neutral objectives (heralds/grubs), macro play, and elder drake control

**Model Development:**
- Implemented 6 classification models: Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Bagging
- Hyperparameter tuning using GridSearchCV with 5-fold cross-validation
- Baseline comparison and comprehensive evaluation metrics

**Evaluation:**
- Comprehensive metrics: accuracy, precision, recall, F1, ROC-AUC, log loss
- Baseline comparison against majority class prediction
- Visualization: confusion matrices, ROC curves, precision-recall curves

## Results

### Model Performance

All models achieved exceptional performance, with top performers reaching 95.5% accuracy and 99.3% ROC-AUC:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Log Loss | Best CV F1 |
|-------|----------|-----------|--------|----------|---------|----------|------------|
| **Logistic Regression** | **0.9557** | **0.9524** | **0.9576** | **0.9550** | **0.9930** | **0.1082** | **0.9536** |
| **Gradient Boosting** | **0.9557** | **0.9524** | **0.9576** | **0.9550** | **0.9924** | 0.1177 | 0.9530 |
| Bagging | 0.9540 | 0.9489 | 0.9576 | 0.9532 | 0.9919 | 0.1880 | 0.9534 |
| Random Forest | 0.9522 | 0.9450 | 0.9582 | 0.9516 | 0.9917 | 0.1249 | 0.9528 |
| Decision Tree | 0.9403 | 0.9239 | 0.9570 | 0.9402 | 0.9771 | 0.1712 | 0.9420 |
| Naive Bayes | 0.9349 | 0.9267 | 0.9418 | 0.9342 | 0.9839 | 0.1617 | N/A |

**Baseline accuracy:** 50.24% (majority class prediction)

### Analysis

#### Model Performance Insights

1. **Logistic Regression as Top Performer**: Surprisingly, the simplest linear model tied with Gradient Boosting for best performance. This indicates that the PCA-transformed feature space exhibits strong **linear separability** - the 9 principal components create a representation where a hyperplane can effectively separate wins from losses. The lowest log loss (0.1082) also shows it produces the most calibrated probability estimates.

2. **Ensemble Methods Cluster Tightly**: Gradient Boosting, Bagging, and Random Forest all achieve 95.2-95.6% F1 scores with 99.2%+ ROC-AUC. The consistency across different ensemble approaches validates that the patterns in the data are robust and not artifacts of a specific modeling technique.

3. **High ROC-AUC Across All Models**: Even the simplest Naive Bayes achieves 98.4% ROC-AUC, demonstrating that all models have excellent ability to rank matches by win probability. The near-perfect discrimination (99.3% for top models) suggests the 9 PCA components capture nearly all predictive signal in the original 60+ features.

4. **Decision Tree Baseline**: The single Decision Tree (94.0% F1) establishes that even without ensembling, tree-based methods can learn the nonlinear decision boundaries. The 5.5 percentage point gap to Random Forest shows the value of variance reduction through bagging.

#### Cross-Validation Consistency

The "Best CV F1" scores (from 5-fold cross-validation during hyperparameter tuning) closely match test set performance (within 0.2%), indicating:
- **No overfitting**: Models generalize well to unseen matches
- **Stable performance**: Results are reliable across different data splits
- **Proper validation**: The train-test split accurately represents expected real-world performance

### PCA Component Analysis

The 9 principal components reveal what factors drive match outcomes in professional League of Legends:

#### **PC1: Gold Advantage and Towers**
**Top Features:** `gpr`, `gspd`, `golddiffat20`, `towers`, `golddiffat25`

The strongest component captures economic dominance. Gold per minute ratio (`gpr`), gold differential at 20 and 25 minutes, and tower control form the foundation of winning. This aligns with the competitive meta - teams that secure gold leads in mid-game typically convert them to objectives and victory.

#### **PC2: Vision Control**
**Top Features:** `visionscore`, `wardsplaced`, `wardskilled`, `gamelength`, `controlwardsbought`

Vision dominance is the second most important factor. High vision scores, ward placement/clearing, and control ward investment indicate map control. `gamelength` appears here because longer games amplify vision metrics - teams that maintain vision control in extended games typically win.

#### **PC3: Combat Metrics**
**Top Features:** `damagetochampions`, `damagetakenperminute`, `teamdeaths`, `ckpm`, `deathsat15`

Teamfight effectiveness and early skirmishing. Damage output/absorption, kill participation (`ckpm`), and limiting early deaths (`deathsat15`) reflect mechanical skill and fight execution. Teams that win fights efficiently tend to snowball advantages.

#### **PC4: Grubs and Herald**
**Top Features:** `opp_void_grubs`, `void_grubs`, `heralds`, `firstherald`, `opp_heralds`

Early neutral objective control. Void grubs (introduced in 2024) and Rift Herald are critical for establishing map pressure before 20 minutes. Securing these objectives provides tempo advantages that enable tower taking and vision control.

#### **PC5: Herald Control**
**Top Features:** `opp_heralds`, `heralds`, `firstherald`, `wpm`, `wcpm`

A refined view of Herald priority specifically, combined with ward metrics. Teams that prioritize herald AND maintain vision control around it demonstrate superior early-game macro coordination.

#### **PC6: Herald and Grubs**
**Top Features:** `heralds`, `firstherald`, `opp_heralds`, `void_grubs`, `opp_void_grubs`

**Note:** PC4, PC5, and PC6 all heavily load on herald/grubs, indicating these objectives appear in multiple orthogonal dimensions of the data. This redundancy suggests that **early neutral objective control is multifaceted** - timing (first herald), total count, and denying opponents all contribute independently to match outcomes.

#### **PC7: Laning Phase and Early Baron Control**
**Top Features:** `firstdragon`, `opp_void_grubs`, `void_grubs`, `opp_barons`, `deathsat10`

Objective sequencing and early game execution. First dragon priority, grub control, and limiting deaths at 10 minutes reflect early macro strategy. The inclusion of `opp_barons` suggests teams with poor early game also struggle to contest Baron later.

#### **PC8: Early Macro and Objective Priority**
**Top Features:** `firsttower`, `dragons`, `elders`, `opp_dragons`, `firstdragon`

The objective control continuum from early (first tower, first dragon) through mid-game (dragon stacking) to late-game (elder drakes). Teams that consistently prioritize objectives across all game phases demonstrate superior macro understanding.

#### **PC9: Elder Drake Control**
**Top Features:** `elders`, `opp_elders`, `damagetakenperminute`, `wpm`, `killsat10`

Late-game teamfight and objective control. Elder Drake is the single most impactful late-game objective, and this component captures which teams successfully navigate to elder soul or deny it from opponents. Interestingly, `killsat10` suggests early skirmish success predicts late-game elder access.

### Key Insights

1. **Gold > Everything**: PC1 (gold advantage) being the strongest component confirms the competitive mantra: "gold wins games." Economic leads at 20-25 minutes are the single best predictor of victory.

2. **Vision is Power**: PC2 demonstrates that vision control is nearly as important as gold. Professional teams that dominate vision create opportunities for objective control and avoid unfavorable fights.

3. **Herald/Grubs Meta**: Three components (PC4, PC5, PC6) focus on early neutral objectives, reflecting the 2024 meta where void grubs (new addition) and Rift Herald create significant early-game value. This multi-dimensional representation suggests these objectives influence matches through multiple pathways (tempo, gold, map pressure).

4. **Objective Sequencing Matters**: PC8 shows that teams which prioritize objectives at ALL game phases (towers → dragons → elders) have the highest win rates. Consistent macro play across the entire match timeline is critical.

5. **Convergent Strategy**: The high model performance (95.5%) despite different algorithms suggests that winning patterns in professional LoL are **consistent and learnable**. The factors that separate winners from losers are not random - they follow predictable patterns centered on gold generation, objective control, and vision dominance.

6. **Linear Separability**: Logistic Regression matching Gradient Boosting indicates that after PCA transformation, the decision boundary is approximately linear. This means the 9 components create a feature space where "more is better" across all dimensions - there are minimal complex interactions or non-monotonic relationships.

## Tech Stack

**Languages:** Python 3.9+

**Core Libraries:**
- Data: pandas, NumPy
- ML: scikit-learn, XGBoost
- Visualization: matplotlib, seaborn
- Experiment Tracking: MLflow
- Model Interpretation: SHAP

**Tools:** Jupyter Notebooks, Git, pip/virtualenv

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Personal_Project
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. Download data from [Oracle's Elixir](https://oracleselixir.com/tools/downloads) and place in `data/raw/`

## Usage

**Run complete pipeline:**
```bash
# Preprocess data
python scripts/run_preprocessing.py

# Create PCA features
python scripts/run_feature_engineering.py

# Train all models
python scripts/train_all_models.py

```

**Train individual model:**
```bash
# Train a specific model
python scripts/train_single_model.py gradient_boosting

# With hyperparameter tuning
python scripts/train_single_model.py logistic_regression

# Enable MLflow tracking
python scripts/train_all_models.py --mlflow
mlflow ui
```

**Use in Python:**
```python
from src.data import load_processed_data
from src.models import train_model

data = load_processed_data('pca')
model = train_model('gradient_boosting', data=data)
```

## What I Learned

This project taught me how to structure a real ML project beyond just notebooks:
- Modular code design - separating data, features, models, and evaluation
- Experiment tracking - using MLflow to log all runs and compare models
- Proper evaluation - avoiding data leakage, using appropriate metrics, baseline comparisons
- Statistical validation - permutation testing to ensure PCA components are meaningful
- Pipeline automation - scripts to reproduce the entire workflow

The biggest technical challenge was preventing data leakage when scaling features - ensuring the scaler is fit only on training data. I also learned that log loss requires probability predictions, not binary outputs.

## Future Improvements

Things I'd like to add:
- Temporal validation (train on early 2024, test on Worlds)
- Model deployment with FastAPI
- SHAP analysis for better interpretability
- Dashboard for live match prediction
- Meta-shift detection across game patches

## Data Source

Dataset: [Oracle's Elixir - 2024 LoL Esports Match Data](https://oracleselixir.com/tools/downloads)

Contains comprehensive match statistics from professional League of Legends tournaments including regional leagues and international events.

## Acknowledgments

Thanks to Oracle's Elixir for maintaining high-quality esports data that made this project possible.
