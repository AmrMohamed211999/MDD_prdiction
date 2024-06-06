import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
rf_model = joblib.load('rf_model.pkl')

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Streamlit app
st.title("RNA-Seq Depression Prediction")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data)
    predictions = rf_model.predict(data)
    result = ["Depressed" if pred == 1 else "Healthy" for pred in predictions]
    st.write("Predictions:")
    st.write(result)
    if 'label' in data.columns:
        from sklearn.metrics import confusion_matrix
        y_test = data['label']
        y_pred = predictions
        cm = confusion_matrix(y_test, y_pred)
        class_names = ['Healthy', 'Depressed']
        plot_confusion_matrix(cm, class_names)
    else:
        st.write("No true labels provided; unable to compute confusion matrix.")

