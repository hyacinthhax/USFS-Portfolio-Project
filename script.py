import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.compose import ColumnTransformer

# Load the data
data = pd.read_csv('cover_data.csv')

# Extract features and labels
X = data.drop('class', axis=1)
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify columns to skip one-hot encoding
binary_columns = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']
integer_columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 
                   'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                   'Horizontal_Distance_To_Fire_Points']

# Identify the remaining columns for one-hot encoding
remaining_columns = list(set(X.columns) - set(binary_columns) - set(integer_columns))

# Define preprocessing for numerical and categorical features
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, integer_columns),
        ('remaining', 'passthrough', remaining_columns)
    ])

# Preprocess the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Encode cover types to integers
cover_types = y.unique()
cover_type_mapping = {cover_type: idx for idx, cover_type in enumerate(cover_types)}
y_train_encoded = y_train.map(cover_type_mapping)
y_test_encoded = y_test.map(cover_type_mapping)

# Build the model
num_classes = len(cover_types)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_preprocessed.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_preprocessed, y_train_encoded, epochs=10)

# Evaluate the model
accuracy = model.evaluate(X_test_preprocessed, y_test_encoded)[1]
print(f'Model Accuracy: {accuracy}')
