import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error


@st.cache
def load_data():
    return pd.read_csv("Orange_Quality_Data.csv")


st.title("Klasifikasi Kualitas Jeruk 🍊")
st.write("Aplikasi ini memprediksi kualitas jeruk menggunakan berbagai algoritma machine learning.")


data = load_data()
st.write("### Dataset")
st.dataframe(data.head())


features = data.columns[:-1]  
target = data.columns[-1]     


X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model_option = st.selectbox(
    "Pilih model yang ingin digunakan:",
    ("Random Forest", "SVM", "Neural Network", "Naive Bayes")
)

if model_option == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif model_option == "SVM":
    model = SVC(kernel='rbf', random_state=42)
elif model_option == "Neural Network":
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
elif model_option == "Naive Bayes":
    model = GaussianNB()


model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5

st.write(f"### Hasil Evaluasi ({model_option})")
st.write(f"- **Akurasi**: {accuracy:.2f}")
st.write(f"- **RMSE**: {rmse:.2f}")
