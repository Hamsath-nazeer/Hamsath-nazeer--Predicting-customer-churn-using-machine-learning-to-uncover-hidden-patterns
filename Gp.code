# app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="🌸 Iris Classifier", layout="wide")
st.title("🌸 Iris Species Classification App")

# Load dataset
@st.cache_data
def load_data():
    df = sns.load_dataset('iris')
    df.dropna(inplace=True)
    return df

df = load_data()

st.subheader("🔍 Dataset Preview")
st.dataframe(df)

# EDA toggle
if st.checkbox("📊 Show Pairplot (EDA)"):
    st.subheader("Pairplot")
    fig = sns.pairplot(df, hue='species')
    st.pyplot(fig)

# Feature & Target split
X = df.drop('species', axis=1)
y = df['species']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluation
st.subheader("📈 Model Evaluation")
st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Feature Importance
st.subheader("🌟 Feature Importances")
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
fig2, ax2 = plt.subplots()
feature_importances.sort_values().plot(kind='barh', ax=ax2)
st.pyplot(fig2)
