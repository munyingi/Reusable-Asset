# Data Science Best Practices - Sample

A comprehensive, production-ready data science project template demonstrating industry best practices, reusable workflows, and the Cookiecutter Data Science structure.

## 🎯 Overview

This repository provides **executable Jupyter notebooks** and **reusable code modules** for building scalable data science projects. It follows the **Cookiecutter Data Science** framework and demonstrates techniques that drive **80% of model performance** through proper feature engineering, documentation, and project organization.

### Key Features

- ✅ **Complete ML Pipeline**: From data exploration to model deployment
- ✅ **Executable Notebooks**: Ready-to-run Jupyter notebooks with sample data
- ✅ **Best Practices**: Industry-standard project structure and workflows
- ✅ **Business Impact Focus**: ROI analysis and business metric translation
- ✅ **Comprehensive Documentation**: Model cards, API docs, and guides

## 📊 Project Structure

```
data-science-examples/
├── data/
│   ├── raw/              # Original, immutable data
│   ├── interim/          # Intermediate transformations
│   ├── processed/        # Final datasets for modeling
│   └── external/         # Third-party sources
├── notebooks/            # Jupyter notebooks for exploration
│   ├── 01-data-exploration-eda.ipynb
│   ├── 02-feature-engineering.ipynb
│   └── 03-model-training-evaluation.ipynb
├── src/                  # Source code for production
│   ├── data/             # Data loading and processing
│   ├── features/         # Feature engineering
│   ├── models/           # Training and prediction
│   └── visualization/    # Plotting and reporting
├── models/               # Trained and serialized models
├── reports/              # Generated analysis and figures
├── docs/                 # Project documentation
│   ├── project_documentation_template.md
│   └── model_card_template.md
└── tests/                # Unit and integration tests
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip or conda package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/data-science-examples.git
cd data-science-examples

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Running the Notebooks

The notebooks are designed to run sequentially:

1. **01-data-exploration-eda.ipynb**: Load data, perform EDA, identify patterns
2. **02-feature-engineering.ipynb**: Create interaction, temporal, and domain-specific features
3. **03-model-training-evaluation.ipynb**: Train models, evaluate performance, calculate business impact

Each notebook generates sample data if source files don't exist, so you can run them immediately without external datasets.

## 📓 Notebook Descriptions

### 1. Data Exploration and EDA

**Purpose**: Understand data quality, distributions, and relationships

**Key Sections**:
- Data loading and quality assessment
- Descriptive statistics and distributions
- Target variable analysis (churn rate: ~27%)
- Correlation analysis
- Feature relationships with target
- Categorical feature analysis
- Key insights summary

**Outputs**: 
- Explored dataset saved to `data/interim/`
- Visualization plots for distributions and relationships

---

### 2. Feature Engineering

**Purpose**: Create features that drive 80% of model performance

**Key Sections**:
- **Interaction Features**: Tenure-based ratios, spending patterns, customer value segments
- **Temporal Features**: Customer lifecycle stages, tenure groups, age groups
- **Domain-Specific Features**: Contract commitment scores, payment reliability, service adoption
- **Encoding**: One-hot encoding for categorical variables
- **Scaling**: StandardScaler for numerical features
- **Feature Selection**: ANOVA F-test and Mutual Information

**Outputs**:
- Full feature set: `data/processed/customer_data_features.csv`
- Selected features: `data/processed/customer_data_selected_features.csv`

**Impact**: Expected 20-50% accuracy improvement over baseline

---

### 3. Model Training and Evaluation

**Purpose**: Train, evaluate, and optimize machine learning models

**Key Sections**:
- Train-test split with stratification
- Class imbalance handling (upsampling)
- Multi-model comparison (Logistic Regression, Random Forest, Gradient Boosting)
- Comprehensive evaluation metrics (ROC AUC, F1, Precision, Recall)
- Confusion matrix analysis
- ROC and Precision-Recall curves
- Feature importance analysis
- **Business Impact Analysis**: ROI calculation, revenue saved, campaign costs
- Hyperparameter tuning with GridSearchCV
- Model serialization for deployment

**Outputs**:
- Trained model: `models/random_forest_model.pkl` (or best model)
- Feature names: `models/feature_names.txt`
- Performance metrics and business impact summary

**Business Metrics**:
- Net Benefit calculation
- ROI percentage
- Revenue saved from retention
- Campaign cost analysis

## 🔧 Source Code Modules

### Data Processing (`src/data/`)

```python
from src.data.make_dataset import load_raw_data, clean_data, save_processed_data
```

- `make_dataset.py`: Load, clean, and save datasets
- Implements data immutability principle (raw data never modified)

### Feature Engineering (`src/features/`)

```python
from src.features.build_features import create_interaction_features, create_temporal_features
```

- `build_features.py`: Reusable feature engineering functions
- Modular design for easy integration into pipelines

### Model Training (`src/models/`)

```python
from src.models.train_model import train_model, evaluate_model, save_model
```

- `train_model.py`: Model training, evaluation, and persistence
- Supports multiple algorithms and hyperparameter tuning

## 📈 Business Impact

This project demonstrates how to translate technical metrics into business value:

| Metric | Technical | Business Translation |
|--------|-----------|---------------------|
| **Accuracy** | 90% | Meaningless without context |
| **True Positives** | 54 customers | $54,000 revenue saved |
| **False Positives** | 20 customers | $1,000 campaign cost |
| **ROI** | N/A | 5,300% return on investment |

### Example Business Impact (from notebook 03)

- **Customers Correctly Identified**: 54 churning customers
- **Revenue Saved**: $54,000 (54 × $1,000 LTV)
- **Campaign Cost**: $3,700 (74 campaigns × $50)
- **Net Benefit**: $50,300
- **ROI**: 1,360%

## 📚 Documentation

### Project Documentation

See [`docs/project_documentation_template.md`](docs/project_documentation_template.md) for comprehensive project documentation including:

- Project overview and objectives
- Data sources and descriptions
- Methodology and approach
- Model architecture and hyperparameters
- Evaluation metrics and results
- Deployment instructions
- Maintenance and monitoring

### Model Card

See [`docs/model_card_template.md`](docs/model_card_template.md) for model-specific documentation including:

- Model details and intended use
- Training data and evaluation data
- Performance metrics across subgroups
- Ethical considerations and limitations
- Bias and fairness assessments

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_features.py
```

## 🔄 Reproducibility

This project ensures reproducibility through:

1. **Data Immutability**: Raw data never modified in place
2. **Version Control**: Git for code, DVC for data (recommended)
3. **Random Seeds**: Fixed seeds (42) for consistent results
4. **Environment Management**: requirements.txt with pinned versions
5. **Documentation**: Comprehensive docs and inline comments

## 📊 Key Insights

### Feature Engineering Impact

- **80%** of model performance comes from feature engineering
- **20-50%** accuracy gains from engineered features
- **2-5%** gains from algorithm optimization alone

### Project Organization Benefits

- **70%** reduction in onboarding time with standardized structure
- **40%** faster project completion with frameworks
- **3×** higher deployment success rates

### Failure Prevention

- **87%** of data science projects fail without proper structure
- **30%** failure rate with best practices implemented
- **60%** of time wasted searching for data without organization

## 🚀 Deployment

### Model Serving

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/random_forest_model.pkl')

# Load feature names
with open('models/feature_names.txt', 'r') as f:
    feature_names = f.read().splitlines()

# Make predictions
def predict_churn(customer_data):
    """Predict churn probability for a customer"""
    X = pd.DataFrame([customer_data], columns=feature_names)
    churn_probability = model.predict_proba(X)[0][1]
    return churn_probability
```

### API Integration

For production deployment, consider:

- **Flask/FastAPI**: REST API for model serving
- **Docker**: Containerization for consistent environments
- **Monitoring**: Track prediction accuracy, data drift, feature distributions
- **Retraining**: Scheduled retraining pipeline (quarterly or on performance degradation)

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Cookiecutter Data Science**: Project structure template
- **Scikit-learn**: Machine learning library
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Visualization

## 📞 Contact

For questions or feedback:

- Open an issue in this repository
- Email: contact@samwelmunyingi.com
- LinkedIn: [Samwel Munyingi](https://www.linkedin.com/in/samwel-munyingi-352a88161
)

## 🔗 Resources

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [Data Science Best Practices](https://www.datascience-pm.com/)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- [ML Ops Best Practices](https://ml-ops.org/)

---

**Built with ❤️ demonstrating data science best practices**
