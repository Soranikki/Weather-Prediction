from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load and prepare the dataset
df = pd.read_csv("seattle-weather.csv")
df.drop(columns=['date'], inplace=True)
mapping = {'drizzle': 0, 'fog': 1, 'rain': 2, 'snow': 3, 'sun': 4}
df['weather'] = df['weather'].map(mapping)

# Features and target
X = df[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = df['weather']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural Network Model
model_Neural_Net = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(5, activation='softmax')
])
model_Neural_Net.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_Neural_Net.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Bagging Model
bagging_lr = BaggingRegressor(estimator=LinearRegression(), n_estimators=10, random_state=42)
bagging_lr.fit(X_train, y_train)

# Stacking Model
estimators = [('lr', LinearRegression()), ('lasso', Lasso())]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stacking_model.fit(X_train, y_train)

# Mapping from numbers to weather descriptions
weather_mapping = {0: 'drizzle', 1: 'fog', 2: 'rain', 3: 'snow', 4: 'sun'}

@app.route("/", methods=["GET", "POST"])
def index():
    nn_output = None
    bagging_output = None
    stacking_output = None
    
    if request.method == "POST":
        # Get user input from the form
        precipitation = float(request.form["precipitation"])
        temp_max = float(request.form["temp_max"])
        temp_min = float(request.form["temp_min"])
        wind = float(request.form["wind"])
        
        # Prepare input for prediction
        input_data = np.array([[precipitation, temp_max, temp_min, wind]])
        scaled_input = scaler.transform(input_data)
        
        # Make predictions
        nn_pred = model_Neural_Net.predict(scaled_input)
        nn_class = np.argmax(nn_pred)
        nn_output = weather_mapping.get(nn_class, 'Unknown')

        bagging_pred = bagging_lr.predict(scaled_input)
        bagging_class = int(round(bagging_pred[0]))  # Rounding to nearest integer for classification
        bagging_output = weather_mapping.get(bagging_class, 'Unknown')

        stacking_pred = stacking_model.predict(scaled_input)
        stacking_class = int(round(stacking_pred[0]))  # Rounding to nearest integer for classification
        stacking_output = weather_mapping.get(stacking_class, 'Unknown')

    return render_template(
        'index.html',
        nn_output=nn_output,
        bagging_output=bagging_output,
        stacking_output=stacking_output
    )

if __name__ == "__main__":
    app.run(debug=False)