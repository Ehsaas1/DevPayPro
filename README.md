DevPayPro – Developer Salary Predictor & Compensation Analysis

DevPayPro is a data-driven project that analyzes global developer compensation trends and predicts salaries using machine learning.
The project is built on the 2023 Stack Overflow Developer Survey and focuses on identifying the key factors that influence developer pay across regions, experience levels, and work styles.

📌 Project Objectives

Analyze long-term developer salary trends

Identify major salary drivers such as experience, education, location, and remote work

Build a machine learning model to predict developer salaries

Provide data-backed insights for compensation analysis

📊 Dataset

Source: Stack Overflow Developer Survey 2023

Records: 70,000+ developer responses

Key Features Used:

Years of coding experience

Education level

Country / region

Employment type

Remote work status

Technology stack

🧹 Data Processing & Feature Engineering

Removed missing and inconsistent records

Handled outliers in salary distributions

Encoded categorical variables

Scaled numerical features

Selected high-impact features based on correlation and variance analysis

🤖 Machine Learning Model

Algorithm: Random Forest Regressor

Baseline Comparison: Linear Regression

Performance Improvements:

~17% reduction in RMSE compared to baseline models

Achieved a realistic R² score of ~0.53

📈 Key Insights

Experience and geography are the strongest predictors of salary

Remote work has a noticeable impact on compensation in certain regions

Salary growth patterns differ significantly across countries and job roles

🛠️ Tech Stack

Programming Language: Python

Libraries & Tools:

Pandas, NumPy

Scikit-learn

Matplotlib / Seaborn

Jupyter Notebook

🚀 How to Run the Project

Clone the repository

git clone https://github.com/Ehsaas1/DevPayPro.git


Install required dependencies

pip install -r requirements.txt


Run the notebook to explore analysis and predictions

📌 Future Improvements

Deploy the model as a web application

Add Power BI / Tableau dashboard for interactive insights

Improve prediction accuracy using hyperparameter tuning

Support salary comparison across multiple years

👤 Author

Ehsaas Choudhary
B.Tech CSE (AIML)
Aspiring Data Analyst / Data Scientist
