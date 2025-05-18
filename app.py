import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# App Title
st.title("Iris Species Prediction App")
st.write("This app predicts the Iris species based on sepal and petal dimensions.")

# Load the Iris dataset
df = sns.load_dataset('iris')

# Data Preprocessing
df.dropna(inplace=True)  # Drop missing values if any

# Data Exploration
if st.checkbox("Show Raw Data"):
    st.write(df.head())

# Feature and target separation
X = df.drop('species', axis=1)
y = df['species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Building
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# User Inputs
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X['sepal_length'].min()), float(X['sepal_length'].max()), float(X['sepal_length'].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X['sepal_width'].min()), float(X['sepal_width'].max()), float(X['sepal_width'].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X['petal_length'].min()), float(X['petal_length'].max()), float(X['petal_length'].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(X['petal_width'].min()), float(X['petal_width'].max()), float(X['petal_width'].mean()))

# Prediction
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)

st.subheader("Prediction")
st.write("Predicted Species: ", prediction[0])

# Model Evaluation
st.subheader("Model Evaluation")
y_pred = model.predict(X_test_scaled)
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Feature Importance
st.subheader("Feature Importance")
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
fig, ax = plt.subplots()
feature_importances.sort_values().plot(kind='barh', ax=ax)
st.pyplot(fig)
