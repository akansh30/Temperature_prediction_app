import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def load_and_check_data(filepath):
    print("Loading data...")
    data = pd.read_csv(filepath)
    print("Data loaded. Checking for missing values...")
    print(data.isnull().sum())  # Check for missing values in each column
    return data

def preprocess_data(data):
    # Handle missing values
    data_cleaned = data.dropna()  # Drop rows with missing values
    
    # Convert non-numeric columns to numeric
    data_cleaned['dt'] = pd.to_datetime(data_cleaned['dt'], errors='coerce')  # Convert dates to datetime
    data_cleaned['Year'] = data_cleaned['dt'].dt.year  # Extract year
    data_cleaned['Month'] = data_cleaned['dt'].dt.month  # Extract month
    data_cleaned['Day'] = data_cleaned['dt'].dt.day  # Extract day
    data_cleaned = data_cleaned.drop(columns=['dt'])  # Drop original date column
    
    # Convert categorical columns to dummy variables if needed (in this case, `Country`)
    data_cleaned = pd.get_dummies(data_cleaned, columns=['Country'], drop_first=True)
    
    return data_cleaned

def train_model():
    # Load and preprocess data
    data = load_and_check_data('data/GlobalLandTemperatures_GlobalLandTemperaturesByCountry.csv')
    data_cleaned = preprocess_data(data)
    
    # Replace 'AverageTemperature' with the actual target column if different
    X = data_cleaned.drop(columns=['AverageTemperature'])  # Features
    y = data_cleaned['AverageTemperature']  # Target variable

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Mean Squared Error: {mse}")

    # Save the model
    joblib.dump(model, 'model/model.pkl')
    print("Model saved to 'model/model.pkl'")

if __name__ == '__main__':
    train_model()
