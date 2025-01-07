# Diabetes Prediction Using Machine Learning

This repository contains a machine learning project aimed at predicting the likelihood of diabetes in individuals based on various health metrics. The project leverages Python and popular machine learning libraries to preprocess data, train models, and evaluate performance.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Installation](#installation)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Diabetes is a chronic medical condition affecting millions worldwide. Early prediction can help in timely treatment and management. This project uses machine learning techniques to analyze patient data and predict whether they are likely to have diabetes.

## Dataset

The dataset used in this project contains health information of individuals, such as:

- Glucose level
- Blood pressure
- BMI (Body Mass Index)
- Age
- Insulin levels

The dataset is preprocessed to handle missing values, normalize features, and ensure it is ready for model training.

## Project Workflow

1. **Data Preprocessing**:
   - Handle missing values.
   - Normalize numerical features.
   - Split data into training and testing sets.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize relationships between features.
   - Understand distributions of key variables.

3. **Model Selection**:
   - Train various machine learning models such as Logistic Regression, Decision Trees, and Random Forests.
   - Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.

4. **Evaluation**:
   - Test the final model on unseen data.
   - Generate confusion matrices and ROC-AUC curves.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Diabetes_Machine_Learning.ipynb
   ```

## Results

The machine learning models achieved the following performance (example values):

- **Logistic Regression**: Accuracy = 78%, F1-Score = 0.75
- **Random Forest**: Accuracy = 85%, F1-Score = 0.82

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Seaborn

## Contributing

Contributions are welcome! Please fork the repository and create a pull request for any improvements or additional features.

## License

This project is licensed under the [MIT License](LICENSE).

---
