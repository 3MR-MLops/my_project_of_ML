Project Overview
​This project predicts passenger survival on the Titanic. It showcases the complete machine learning lifecycle, starting from raw data processing to deploying a high-performance model via an API.
​Workflow & Evolution
​Data Preprocessing: Cleaned the Titanic dataset, handled missing values, and encoded categorical features (like Sex) for model compatibility.
​Model Selection & Optimization:
​Baseline: Started with a simple Decision Tree to understand the basic logic.
​Ensemble: Improved accuracy by implementing a Random Forest (100 trees) to reduce overfitting.
​Final Model: Used XGBoost (Extreme Gradient Boosting) to push the performance further, achieving an accuracy of 83%.
​Deployment: Serialized the final model using Pickle and built a FastAPI web service to provide real-time survival predictions.
