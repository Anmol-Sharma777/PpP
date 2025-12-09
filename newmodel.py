import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib #  joblib for saving the model

df = pd.read_csv('Bangalore_cleaned_selected_columns.csv')

#  Data Preprocessing 

# 1. Binary Feature Transformation: Convert '9' to '1' in identified binary columns
amenity_columns = [
    'MaintenanceStaff', 'Gymnasium', 'SwimmingPool', 'LandscapedGardens', 'JoggingTrack',
    'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall', 'Intercom', 'SportsFacility',
    'ATM', 'ClubHouse', 'School', '24X7Security', 'PowerBackup', 'CarParking',
    'StaffQuarter', 'Cafeteria', 'MultipurposeRoom', 'Hospital', 'WashingMachine',
    'Gasconnection', 'AC', 'Wifi', "Children'splayarea", 'LiftAvailable', 'BED',
    'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV', 'DiningTable', 'Sofa',
    'Wardrobe', 'Stadium'
]

for col in amenity_columns:
    df[col] = df[col].apply(lambda x:1 if x>0 else x)

print(f"Unique values after transformation for 'MaintenanceStaff': {df['MaintenanceStaff'].unique()}")
print(f"Unique values after transformation for 'Wifi': {df['Wifi'].unique()}")

# Target Variable Transformation: Apply log transformation to 'Price'
df['Price_log'] = np.log1p(df['Price'])

#  Feature Engineering - Location Handling (Group less frequent locations)
location_counts = df['Location'].value_counts()
rare_location_threshold = 20
locations_to_keep = location_counts[location_counts >= rare_location_threshold].index.tolist() # Convert to list
print(f"Saving {len(locations_to_keep)} locations to keep.")

# Save locations_to_keep
joblib.dump(locations_to_keep, 'locations_to_keep.pkl')
print("locations_to_keep saved to locations_to_keep.pkl")


df['Location_grouped'] = df['Location'].apply(lambda x: x if x in locations_to_keep else 'Other')
print(f"\nNumber of unique locations after grouping: {df['Location_grouped'].nunique()}")


X = df.drop(['Price', 'Price_log', 'Location'], axis=1) 
y = df['Price_log']

# Identify numerical and categorical features for the preprocessor
numerical_features = ['Area', 'No. of Bedrooms']
categorical_features = ['Location_grouped']
binary_features = [col for col in X.columns if col not in numerical_features and col not in categorical_features]


# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features + binary_features),
        ('cat', categorical_transformer, categorical_features)
    ])

#  Model Application & Hyperparameter Tuning


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for the model with the preprocessor
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', RandomForestRegressor(random_state=42))])

# Define the parameter grid for GridSearchCV
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_features': [0.8, 1.0],
    'regressor__max_depth': [10, 20],
    'regressor__min_samples_split': [5, 10],
}

print("\nStarting GridSearchCV for Hyperparameter Tuning with RandomForestRegressor...")
grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='r2', verbose=1)
grid_search.fit(X_train, y_train)

print("\nGridSearchCV complete.")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation R-squared: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

model_filename = 'bangalore_house_price_model.pkl'
joblib.dump(best_model, model_filename)
print(f"\nModel saved to {model_filename}")

y_pred_log = best_model.predict(X_test)

# Inverse transform the predictions to get actual price scale
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)


mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)

print(f"\nModel Evaluation with Tuned RandomForestRegressor and Grouped Locations:")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R-squared (R2): {r2:.4f}")

predictions_df = pd.DataFrame({
    'Original_Index': X_test.index, 
    'Actual_Price': y_test_original,
    'Predicted_Price': y_pred
})

predictions_filename = 'bangalore_house_price_predictions.csv'
predictions_df.to_csv(predictions_filename, index=False)
print(f"\nPredictions saved to {predictions_filename}")