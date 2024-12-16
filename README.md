# Supply-Chain-Resilience-Ensuring-Seamless-Delivery-During-Disruptions
  

## Overview  
This project analyzes the supply chain for instant noodles using exploratory data analysis (EDA) and machine learning regression models. It identifies key factors affecting production weight (`product_wg_ton`) and predicts output using various models to optimize supply chain efficiency.  

## Objectives  
- Perform EDA to gain insights into supply chain operations.  
- Handle missing values, outliers, and perform feature engineering.  
- Use regression models to predict production weight.  
- Evaluate model performance and select the best-performing model.  

## Dataset  
- **Target Variable**: `product_wg_ton` (Production weight in tons).  
- **Features**: Includes warehouse details, storage issues, temperature regulation, and more.  
- **Size**: Shape of the dataset provided after cleaning (`df1.shape`): Updated in the script.  

## Steps Involved  

### 1. **Data Preprocessing**  
- **Handling Missing Values**:  
  - Numerical columns filled with mean values.  
  - Categorical columns filled with mode or dropped if unnecessary.  
- **Outliers**:  
  - Identified using IQR and capped to minimize skewness.  
- **Feature Engineering**:  
  - Converted categorical columns to numerical using one-hot encoding.  
  - Extracted specific information, such as the last character from `WH_regional_zone`.  

### 2. **Exploratory Data Analysis (EDA)**  
- **Visualization Techniques**:  
  - Correlation heatmaps to identify relationships between features.  
  - Bar plots to explore key factors like storage issues, transportation, and production.  
  - Box plots to detect and analyze outliers.  

### 3. **Model Building**  
- **Regression Models Evaluated**:  
  - Linear Regression  
  - Lasso and Ridge  
  - K-Nearest Neighbors (KNN)  
  - Decision Tree  
  - Random Forest  
  - XGBoost  
  - AdaBoost  
- **Best Model**:  
  - Random Forest achieved the highest R² score.  

### 4. **Evaluation Metrics**  
- **Metrics Used**:  
  - Root Mean Squared Error (RMSE)  
  - Mean Absolute Error (MAE)  
  - R² Score  

## Results  
- **Best Model**: Random Forest Regressor  
  - **Training Set**:  
    - RMSE: Value from script.  
    - R²: Value from script.  
  - **Test Set**:  
    - RMSE: Value from script.  
    - R²: Value from script.  

## Key Insights  
- **Storage Issues and Production**: More storage issues lead to higher production challenges.  
- **Transportation and Production**: Delays in transport reduce production efficiency.  
- **Temperature Regulation**: Effective regulation improves production consistency.  

## Requirements  
Install the required Python libraries:  
```bash  
pip install pandas numpy matplotlib seaborn scikit-learn xgboost  
```  

## How to Run  
1. **Load Dataset**: Replace the file path in the script (`df = pd.read_csv(...)`) with your dataset location.  
2. **Preprocess Data**: Run the preprocessing steps to clean and transform the data.  
3. **Train Models**: Use the provided model training loop to evaluate different algorithms.  
4. **Evaluate Results**: Compare the performance metrics to choose the best model.  
5. **Visualize Results**: Use scatter plots and heatmaps to interpret predictions.  

## Future Enhancements  
- Incorporate additional features like external factors (e.g., weather, raw material availability).  
- Explore advanced machine learning models, such as neural networks.  
- Optimize models using hyperparameter tuning techniques.  
