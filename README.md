# Housing Prices Prediction: Advanced Regression and Ensemble Methods

This project tackles the "Housing Prices Competition for Kaggle Learn Users" competition on Kaggle. The goal is to predict the sale price of residential homes in Ames, Iowa, based on 79 explanatory variables. The project focuses on advanced regression techniques, creative feature engineering, and ensemble methods to achieve a robust and accurate prediction model.

## Project Overview

This project utilizes several key techniques:

* **Custom Transformers:** Implements custom transformers using scikit-learn's `BaseEstimator` and `TransformerMixin` for tailored data preprocessing steps, including data type cleaning, feature engineering, composite feature dropping (to combat multicollinearity) and missing value handling.
* **Pipeline:**  A complete preprocessing pipeline is constructed using scikit-learn's `Pipeline` to streamline the data transformation process. This ensures consistency and reproducibility when applying transformations to both training and testing data.
* **Categorical Encoding:** Employs various encoding techniques based on feature characteristics:
    * Binary Encoding for high-cardinality nominal features (e.g., Neighborhood).
    * One-Hot Encoding for low-cardinality nominal features (e.g., MSZoning).
    * Ordinal Encoding with domain expertise for ordinal features (e.g., OverallQual, OverallCond).
* **Feature Engineering:** Creates composite features like "Total Bathrooms", "Basement Value Index", and "Garage Score" to capture potentially important relationships among existing variables.
* **Advanced Regression Models:** Leverages powerful algorithms like Random Forest, Gradient Boosting, and XGBoost as base models.
* **Ensemble Methods:** Combines the base models using ensemble techniques like Voting and Stacking to improve overall prediction accuracy and robustness by mitigating model bias, reducing variance, and improving stability. Stacking ensembles use meta-learner, a model to aggregate the base model predictions.

## File Descriptions

* **`housing-price-ensemble.ipynb`:** Notebook containing the initial data exploration, feature engineering analysis and hyperparameter tuning experiments.
* **`housing-prices-submission.ipynb`:**  Notebook with the final pre-processing pipeline, model training and submission generation code. 

## Key Improvements & Findings (from housing-price-ensemble.ipynb)

* **Data Type Handling:** Correcting data types for categorical features initially represented as numerical values (MSSubClass, OverallQual, OverallCond, MoSold).
* **Composite Feature Dropping:** Eliminating features such as 'GrLivArea' and 'TotalBsmtSF' to address multicollinearity with other, more granular features. 
* **Feature Engineering:** Introducing composite features, notably 'TotalBathrooms', 'BasementValue', and 'GarageScore', which significantly improved model performance. These enhanced the models' ability to capture the overall quality and value of different aspects of the properties.
* **Missing Value Imputation:** Implemented a neighborhood-based median imputation strategy for 'LotFrontage', proving more effective than simple overall median imputation. 
* **Hyperparameter Tuning:** Extensive hyperparameter tuning using RandomizedSearchCV identified optimal parameter sets for each base model (RandomForest, GradientBoosting, XGBoost), leading to a notable improvement in their respective RMSE scores.
* **Ensemble Approach:** The project explores various ensemble methods, including simple averaging, weighted averaging based on model performance, stacking with Ridge regression as the meta-learner and stacking with ElasticNet.  The stacking ensemble with Ridge as the meta-learner achieved the best performance, confirming the benefit of combining model predictions.

## Instructions

1. **Download the data:**  Get the competition data from [Kaggle](https://www.kaggle.com/c/home-data-for-ml-course/data).
2. **Run the notebooks:** Execute `housing-price-ensemble.ipynb` first for detailed data exploration, feature engineering justification, and tuning experiments. Then, run `housing-prices-submission.ipynb` to generate the submission file. This notebook includes a robust preprocessing pipeline that handles categorical encoding and missing values effectively.


## Results

The best-performing model, a stacking ensemble of Gradient Boosting and XGBoost with Ridge as meta-learner, achieved an RMSE of approximately 28,004 on cross-validation, demonstrating a substantial improvement over the baseline model. This suggests that stacking with meta-learner can increase the overall predictability of the models by leveraging their individual strengths.

## Further Improvements

* **Further Feature Engineering:** Exploring more sophisticated feature interactions and transformations, potentially incorporating domain expertise regarding real estate valuation.
* **Meta-Learner Optimization:** Fine-tuning the meta-learner model by adjusting hyperparameters. 
* **Stacking with diverse learners:** Experimenting with different meta-learners beyond Ridge regression, like support vector regression or neural networks.

This project provides a robust foundation for predicting housing prices and demonstrates the power of thoughtful feature engineering, model selection and ensemble approaches.  
