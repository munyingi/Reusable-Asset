# Deployment Guide

This guide explains how to deploy this data science repository to GitHub and use it as a template for your projects.

## 📦 Deploying to GitHub

### Option 1: Create a New Repository

1. **Create a new repository on GitHub**:
   - Go to https://github.com/new
   - Name your repository (e.g., `data-science-best-practices`)
   - Choose public or private
   - Do NOT initialize with README, .gitignore, or license

2. **Push this repository to GitHub**:
   ```bash
   cd data-science-examples
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git branch -M main
   git push -u origin main
   ```

### Option 2: Fork and Clone

1. **Fork this repository** on GitHub (if available as a template)
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/your-fork-name.git
   cd your-fork-name
   ```

## 🚀 Using as a Template

### For New Projects

1. **Copy the structure**:
   ```bash
   cp -r data-science-examples my-new-project
   cd my-new-project
   rm -rf .git
   git init
   ```

2. **Customize for your project**:
   - Update `README.md` with your project details
   - Modify notebooks for your specific use case
   - Update `requirements.txt` with your dependencies
   - Replace sample data with your actual datasets

3. **Initialize new Git repository**:
   ```bash
   git add .
   git commit -m "Initial commit from template"
   git remote add origin https://github.com/yourusername/my-new-project.git
   git push -u origin main
   ```

## 🔧 Configuration

### Environment Setup

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import pandas, numpy, sklearn; print('All packages installed successfully')"
   ```

### Jupyter Configuration

1. **Install Jupyter kernel**:
   ```bash
   python -m ipykernel install --user --name=data-science-env
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

3. **Select kernel**: In Jupyter, select `Kernel > Change Kernel > data-science-env`

## 📊 Data Management

### Using Your Own Data

1. **Place raw data** in `data/raw/`:
   ```bash
   cp /path/to/your/data.csv data/raw/
   ```

2. **Update notebooks** to load your data:
   ```python
   data = pd.read_csv('../data/raw/your_data.csv')
   ```

3. **Never commit raw data** (already in `.gitignore`):
   ```
   data/raw/*
   data/interim/*
   data/processed/*
   ```

### Data Version Control (Optional)

For large datasets, use DVC (Data Version Control):

```bash
pip install dvc
dvc init
dvc add data/raw/large_dataset.csv
git add data/raw/large_dataset.csv.dvc .dvc/config
git commit -m "Add dataset with DVC"
```

## 🧪 Testing Setup

1. **Install testing dependencies**:
   ```bash
   pip install pytest pytest-cov
   ```

2. **Run tests**:
   ```bash
   pytest tests/
   ```

3. **Generate coverage report**:
   ```bash
   pytest --cov=src --cov-report=html tests/
   ```

## 🐳 Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

Build and run:

```bash
docker build -t data-science-project .
docker run -p 8888:8888 -v $(pwd):/app data-science-project
```

## 🔄 CI/CD Setup

### GitHub Actions

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=src
```

## 📝 Documentation Deployment

### GitHub Pages

1. **Enable GitHub Pages** in repository settings
2. **Select source**: `main` branch, `/docs` folder
3. **Access documentation** at: `https://yourusername.github.io/your-repo-name/`

### ReadTheDocs (Optional)

1. **Create `docs/` directory** with Sphinx configuration
2. **Connect repository** to ReadTheDocs
3. **Auto-build** on every commit

## 🔐 Security Best Practices

### Secrets Management

1. **Never commit credentials**:
   - Add `.env` to `.gitignore`
   - Use environment variables
   - Use secret management tools (AWS Secrets Manager, Azure Key Vault)

2. **Example `.env` file**:
   ```
   DATABASE_URL=postgresql://user:pass@localhost/db
   API_KEY=your_api_key_here
   ```

3. **Load secrets in code**:
   ```python
   from dotenv import load_dotenv
   import os
   
   load_dotenv()
   api_key = os.getenv('API_KEY')
   ```

### Dependency Security

1. **Scan for vulnerabilities**:
   ```bash
   pip install safety
   safety check
   ```

2. **Keep dependencies updated**:
   ```bash
   pip list --outdated
   pip install --upgrade package_name
   ```

## 🌐 Sharing and Collaboration

### Making it Public

1. **Clean sensitive data**:
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch path/to/sensitive/file" \
     --prune-empty --tag-name-filter cat -- --all
   ```

2. **Update repository visibility** in GitHub settings

3. **Add collaboration guidelines** in `CONTRIBUTING.md`

### Creating a Template Repository

1. Go to repository **Settings**
2. Check **Template repository**
3. Users can now click **Use this template** to create their own copy

## 📊 Monitoring and Maintenance

### Model Monitoring

1. **Track predictions**:
   ```python
   import mlflow
   
   mlflow.log_metric("accuracy", accuracy)
   mlflow.log_param("model_type", "random_forest")
   ```

2. **Set up alerts** for:
   - Model performance degradation
   - Data drift
   - Prediction latency

### Scheduled Retraining

1. **GitHub Actions scheduled workflow**:
   ```yaml
   on:
     schedule:
       - cron: '0 0 1 * *'  # Monthly on the 1st
   ```

2. **Automated retraining pipeline**:
   - Load latest data
   - Retrain model
   - Evaluate performance
   - Deploy if improved

## 🆘 Troubleshooting

### Common Issues

1. **Import errors**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

2. **Jupyter kernel not found**:
   ```bash
   python -m ipykernel install --user --name=data-science-env
   ```

3. **Git LFS issues with large files**:
   ```bash
   git lfs install
   git lfs track "*.csv"
   git add .gitattributes
   ```

## 📞 Support

For issues or questions:

- **Open an issue** in the GitHub repository
- **Check documentation** in `docs/` directory
- **Review notebook comments** for implementation details

---

**Ready to deploy!** Follow the steps above to get your data science project on GitHub and start collaborating.
