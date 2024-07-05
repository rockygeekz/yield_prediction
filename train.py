from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

def train_model(input_file):
    # Read the dataset
    df = pd.read_csv(input_file)

    # Handle missing values in the target variable 'Production' using SimpleImputer
    imputer = SimpleImputer(strategy='median')
    df['Production'] = imputer.fit_transform(df[['Production']])

    # Prepare features and target
    X = df[['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop', 'Area']]
    y = df['Production']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define columns to be one-hot encoded and scaled
    categorical_cols = ['State_Name', 'District_Name', 'Crop']
    numeric_cols = ['Crop_Year', 'Area']

    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Initialize model (SGDRegressor with incremental learning)
    model = SGDRegressor(max_iter=1000, tol=1e-3)

    # Create full pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

    # Train the model incrementally with progress indicator
    batch_size = 1000
    for batch_start in range(0, len(X_train), batch_size):
        batch_end = min(batch_start + batch_size, len(X_train))
        X_batch = X_train.iloc[batch_start:batch_end]
        y_batch = y_train.iloc[batch_start:batch_end]
        
        # Fit the model on the batch
        pipeline.fit(X_batch, y_batch)
        
        # Calculate current progress
        progress = batch_end / len(X_train) * 100
        print(f"Training progress: {progress:.2f}%")

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Save the model
    model_filename = 'models/crop_yield_model.pkl'
    joblib.dump(pipeline, model_filename)

if __name__ == "__main__":
    input_file = 'data/crop_yield_data_fixed.csv'
    train_model(input_file)
