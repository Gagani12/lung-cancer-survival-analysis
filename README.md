# 🫁 Lung Cancer Survival Prediction

### Identifying Factors Influencing Lung Cancer Patient Survival Using Statistical Analysis and Predictive Modeling

<p align="center">
  <b>A Data Science Project | NSBM Green University</b>
</p>

---

## 📌 Project Overview

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Understanding the factors that influence patient survival is important for improving healthcare decisions and treatment planning.

This project focuses on analyzing lung cancer patient data using statistical analysis and machine learning techniques to identify important factors associated with patient survival and predict survival outcomes.

The analysis explores demographic, lifestyle, clinical, and treatment-related factors to discover meaningful patterns and relationships within the dataset.

---

## 🎯 Project Objectives

- Perform data cleaning and preprocessing.
- Explore patient characteristics through Exploratory Data Analysis (EDA).
- Identify relationships between patient factors and survival outcomes.
- Conduct statistical hypothesis testing.
- Test statistical assumptions before applying analytical models.
- Apply Linear and Multiple Regression techniques.
- Perform dimensionality reduction using PCA.
- Explore clustering patterns in patient data.
- Analyze temporal trends where applicable.
- Develop machine learning classification models.
- Handle class imbalance problems.
- Evaluate and compare predictive models.
- Identify important features influencing survival outcomes.

---

## 📂 Project Structure

```text
Lung-Cancer-Project/
│
├── data/
│   └── lung_cancer.csv
│
├── notebooks/
│   ├── 01_Data_Cleaning.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   ├── 03_Hypothesis_Testing.ipynb
│   ├── 04_Assumption_Testing.ipynb
│   ├── 05_Regression_Analysis.ipynb
│   ├── 06_PCA_Clustering.ipynb
│   ├── 07_Time_Series_Analysis.ipynb
│   ├── 08_Confidence_Intervals_Model_Validation.ipynb
│   ├── 09_Machine_Learning_Classification.ipynb
│   ├── 10_Class_Imbalance_Handling.ipynb
│   ├── 11_Model_Evaluation.ipynb
│   └── 12_Feature_Importance_Business_Interpretation.ipynb
│
├── results/
│   ├── figures/
│   └── tables/
│
├── README.md
└── requirements.txt
```

---

## 🔬 Project Workflow

The project follows a structured Data Science workflow.

| Stage | Analysis |
|---|---|
| 01 | Data Cleaning |
| 02 | Exploratory Data Analysis |
| 03 | Hypothesis Testing |
| 04 | Assumption Testing |
| 05 | Regression Analysis |
| 06 | PCA & Clustering |
| 07 | Time Series Analysis |
| 08 | Confidence Intervals & Model Validation |
| 09 | Machine Learning Classification |
| 10 | Class Imbalance Handling |
| 11 | Model Evaluation |
| 12 | Feature Importance & Business Interpretation |

---

## 📊 Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the dataset and identify important patterns.

The analysis includes:

- Patient age distribution.
- Gender distribution.
- Cancer stage distribution.
- Smoking history.
- Treatment-related patterns.
- Survival outcome distribution.
- Correlation between numerical variables.

### Data Visualizations

The following visualizations are generated during the analysis.

**1. Age Distribution**

!["11)Age_qqplot.png"](results/figures/"C:\Users\hp\Downloads\11)Age_qqplot.png")

**2. 	Regression Analysis Plot**

![Regression Analysis Plot](results/figures/"C:\Users\hp\Downloads\16)regression_residual_plot.png")

**3. Correlation Heatmap**

![Correlation Heatmap](results/figures/"C:\Users\hp\Downloads\10)correlation_heatmap.png")

**4. Confidence Interval**

![Confidence Interval](results/figures/"C:\Users\hp\Downloads\29)confidence_interval_plot.png")

**5. Correlation Heatmap**

![Correlation Heatmap](results/figures/correlation_heatmap.png)

---

## 📈 Statistical Analysis

Statistical techniques are applied to investigate relationships between patient characteristics and survival outcomes.

The statistical analysis includes:

- Descriptive Statistics
- Hypothesis Testing
- Statistical Assumption Testing
- Confidence Interval Estimation
- Correlation Analysis
- Linear Regression
- Multiple Regression

These techniques help identify statistically meaningful relationships within the dataset.

---

## 🧠 Dimensionality Reduction & Clustering

Principal Component Analysis (PCA) is used to reduce the dimensionality of the dataset while preserving important information.

Clustering techniques are explored to identify groups of patients with similar characteristics.

---

## 🤖 Machine Learning

Machine learning classification techniques are applied to predict patient survival outcomes.

The machine learning workflow includes:

- Data Preprocessing
- Feature Selection
- Model Training
- Class Imbalance Handling
- Model Validation
- Performance Evaluation

### Model Evaluation

Models are evaluated using appropriate performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- AUC Score

### Model Performance Visualizations

**1. Confusion Matrix**

![Confusion Matrix](results/figures/confusion_matrix.png)

**2. ROC Curve**

![ROC Curve](results/figures/roc_curve.png)

**3. Model Comparison**

![Model Comparison](results/figures/model_comparison.png)

**4. Feature Importance**

![Feature Importance](results/figures/feature_importance.png)

---

## 📁 Results & Outputs

All generated outputs are organized inside the `results/` directory.

| Folder | Description |
|---|---|
| `results/figures/` | Graphs, charts, and visualizations |
| `results/tables/` | Statistical results and model performance tables |

---

## 🛠️ Technologies & Tools

The following technologies and libraries are used throughout the project.

| Technology | Purpose |
|---|---|
| Python | Programming and data analysis |
| Jupyter Notebook | Interactive analysis |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| SciPy | Statistical analysis |
| Statsmodels | Statistical modeling |

---

## 🚀 Installation & Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone <repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd Lung-Cancer-Project
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Run the Notebooks

Open the `notebooks/` folder and execute the notebooks in the recommended sequence.

---

## 📊 Expected Outcomes

- Identify important factors associated with lung cancer patient survival.
- Understand relationships between demographic and clinical variables.
- Discover meaningful patterns through statistical analysis.
- Develop predictive classification models.
- Compare model performance using appropriate metrics.
- Identify influential features.
- Generate data-driven insights to support healthcare-related research.

---

## 👥 Project Team

This project is collaboratively developed by undergraduate Data Science students at NSBM Green University.

| Team Member | Responsibilities |
|---|---|
| Team Member 1 | Data Cleaning, EDA, Hypothesis Testing & Assumption Testing |
| Team Member 2 | Regression Analysis, PCA & Clustering, Time Series Analysis, Confidence Intervals & Model Validation |
| Team Member 3 | Machine Learning Classification, Class Imbalance Handling, Model Evaluation & Feature Importance |

---

## 🎓 Academic Information

**University:** NSBM Green University  
**Degree:** BSc (Hons) in Data Science  
**Department:** Department of Computer and Data Science  
**Project:** Lung Cancer Survival Prediction

---

## ⚠️ Disclaimer

This project is developed for academic and educational purposes only.

The predictive models and statistical findings are not intended to replace professional medical advice, clinical diagnosis, or treatment decisions.

---

## ⭐ Conclusion

This project demonstrates the application of statistical analysis and machine learning techniques to investigate factors associated with lung cancer patient survival.

By combining data preprocessing, exploratory analysis, statistical methods, predictive modeling, and model interpretation, the project aims to discover meaningful insights from healthcare data.

---

<p align="center">
  <b>Exploring Healthcare Data Through the Power of Data Science 🫁📊</b>
</p>
