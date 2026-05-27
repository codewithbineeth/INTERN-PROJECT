# Data Science Intern Project - CodVeda Internship

Welcome to my complete Data Science portfolio repository, developed during my internship journey with **CodVeda Technology**. This repository showcases end-to-end data pipelines spanning fundamental data collection, intermediate predictive modeling, unsupervised machine learning, and advanced deep learning frameworks.

Over the course of this curriculum, I successfully code-verified and delivered **all 8 tasks** across all programmatic levels.

## 📂 Repository Architecture & Task Index

### 🟢 Level 1: Basic Phase
* **Task 1: Data Collection & Web Scraping**
  * *Objective:* Build an automated multi-page scraper to safely navigate and harvest unstructured web targets.
  * *Implementation:* Utilized `requests` and `BeautifulSoup4` to crawl clean web assets, bypass network blocks via custom User-Agent mapping, and continuously handle multi-page layout pagination loops.
  * *Deliverables:* `level1-basic/task1/task1_scraper.py`, `scraped_quotes.csv`, `scraped_quotes.json`
* **Task 2: Data Cleaning & Preprocessing**
  * *Objective:* Clean and pre-process a raw dataset to eliminate anomalies and make features suitable for algorithmic analysis.
  * *Implementation:* Processed telecom customer churn attributes using `pandas` and `scikit-learn`. Applied median/mode imputation to handle missing rows, isolated and removed mathematical outliers using the Interquartile Range (IQR) filter, encoded binary fields, and normalized vector variance via `StandardScaler`.
  * *Deliverables:* `level1-basic/task2/task2_preprocessing.py`, `cleaned_churn_data.csv`
* **Task 3: Exploratory Data Analysis (EDA)**
  * *Objective:* Compute summary statistics and visualize data trends to understand underlying categorical structure.
  * *Implementation:* Conducted descriptive profiling on the classic Iris dataset. Built multi-panel distributions, feature spreads, and biological cluster tracking grids using `matplotlib` and `seaborn`.
  * *Deliverables:* `level1-basic/task3/task3_eda.py`, `iris_histograms.png`, `iris_boxplots.png`, `iris_scatterplot.png`

### 🔵 Level 2: Intermediate Phase
* **Task 1: Predictive Modeling (Regression)**
  * *Objective:* Evaluate a regression model to forecast a continuous numerical target variable.
  * *Implementation:* Parsed space-separated assets to map neighborhood characteristics against median house values (`MEDV`). Trained and benchmarked Linear Regression, Decision Trees, and Random Forest Regressors using Mean Squared Error (MSE) and $R^2$ variance diagnostics.
  * *Deliverables:* `level2-intermediate/task1/task1_regression.py`, `regression_model_comparison.png`
* **Task 2: Classification Models**
  * *Objective:* Build a classification model to accurately predict a categorical class outcome.
  * *Implementation:* Programmed a pipeline to identify customer churn markers. Benchmarked `LogisticRegression` baselines against non-linear architectures while tracking execution success via deep metric scoring (Accuracy, Precision, Recall) and multi-model ROC Curve profiles.
  * *Deliverables:* `level2-intermediate/task2/task2_classification.py`, `classification_roc_curve.png`
* **Task 3: Unsupervised Clustering**
  * *Objective:* Implement clustering to group unlabelled data points into natural behavioral segments.
  * *Implementation:* Ran distance-based `K-Means` algorithms on user activity parameters. Selected optimal cluster centroids using the Elbow Method (WCSS) and squashed 15-dimensional matrices into a clean 2D plane using Principal Component Analysis (PCA).
  * *Deliverables:* `level2-intermediate/task3/task3_clustering.py`, `clustering_elbow_method.png`, `clustering_pca_segments.png`

### 🔴 Level 3: Advanced Phase
* **Task 1: Time Series Analysis & Forecasting**
  * *Objective:* Model time-series sequence frequencies to forecast future values.
  * *Implementation:* Resampled date index metrics into continuous calendar months using Pandas. Decomposed signals into Trend, Annual Seasonality, and Residual profiles via `statsmodels`. Trained an autoregressive predictive `ARIMA(1,1,1)` model and checked boundary validation via Root Mean Squared Error (RMSE).
  * *Deliverables:* `level3-advance/task1/task1_timeseries.py`, `time_series_decomposition.png`, `time_series_smoothing.png`, `time_series_forecast.png`
* **Task 2: Natural Language Processing (NLP)**
  * *Objective:* Preprocess and classify text data strings into distinct categories.
  * *Implementation:* Built a text analytics pipeline utilizing the Natural Language Toolkit (`nltk`) to lowercase, tokenize, and strip alphanumeric social media posts of non-semantic English stopwords. Transformed cleaned text into sparse numeric matrix vectors via TF-IDF Vectorization and classified them using a regularized model.
  * *Deliverables:* `level3-advance/task2/task2_nlp.py`
* **Task 3: Deep Learning Neural Networks**
  * *Objective:* Design, train, and tune a multi-layer feed-forward neural network classifier.
  * *Implementation:* Implemented a Multi-Layer Perceptron (MLP) Neural Network architecture via `scikit-learn` optimized for Python 3.14 safety. Loaded the MNIST handwritten digits dataset, normalized image pixel grids from a 0-255 scale to a uniform 0.0-1.0 boundary, set hidden ReLU layer parameters, tuned an Adam solver optimizer, and plotted loss convergence graphs over training epochs.
  * *Deliverables:* `level3-advance/task3/task3_neural_network.py`, `neural_network_training_curves.png`

---

## 🛠️ Global Technology Stack & Toolkits
* **Language Core:** Python 3.14
* **Data Management & Web Extraction:** Pandas, NumPy, BeautifulSoup4, Requests
* **Statistical Modeling & ML:** Scikit-Learn (`sklearn`), Statsmodels
* **Data Visualization Graphics:** Matplotlib, Seaborn

---

## 🚀 Environment Setup & Execution

1. **Clone the repository workspace directly onto your computer:**
   ```bash
  git clone [https://github.com/codewithbineeth/INTERN-PROJECT.git](https://github.com/codewithbineeth/INTERN-PROJECT.git)
  cd INTERN-PROJECT

  **Install all required analytics dependencies via pip:  
    python -m pip install pandas numpy beautifulsoup4 requests scikit-learn statsmodels matplotlib seaborn nltk
