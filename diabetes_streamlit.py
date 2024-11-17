import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load the trained model and scaler
with open('diabetes_model.pkl', 'rb') as file:
    classifier = pickle.load(file)

with open('diabetes_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Helper function for prediction
def prediction(input_data):
    input_data = pd.DataFrame(input_data, columns=classifier.feature_names_in_)
    scaled_input_data = scaler.transform(input_data)
    probabilities = classifier.predict_proba(scaled_input_data)
    prediction = classifier.predict(scaled_input_data)
    return prediction[0], probabilities[0]

def main():
    st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

    # Add a home page with an explanation of the project
    home_tab, prediction_tab, eda_tab, feature_engineering_tab, dataset_desc_tab, model_insights_tab = st.tabs([
        "Home", "Prediction", "EDA", "Feature Engineering", "Dataset Description", "Model Insights"
    ])

    with home_tab:
        st.title("Welcome to the Diabetes Prediction App")
        st.markdown(
            """This project aims to predict diabetes using a machine learning model. The app provides detailed insights 
            into the dataset, the feature engineering process, and the model's performance. Whether you're a data enthusiast 
            or a healthcare professional, this app will help you understand the impact of various factors on diabetes prediction."""
        )
        st.image("D:/MachineLearning2/a5qp_qxmg_230817.jpg", use_column_width=True, caption="Understanding Diabetes")

    with prediction_tab:
        st.header("Diabetes Prediction")
        st.sidebar.header("Input Features")
        Pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 1)
        Glucose = st.sidebar.slider('Glucose Level', 50, 200, 100)
        BMI = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
        Age = st.sidebar.slider('Age', 20, 80, 30)
        DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)

        # Derived features
        Glucose_BMI = Glucose * BMI
        Age_Pedigree = Age * DiabetesPedigreeFunction
        AgeGroup_30_40 = 1 if 30 <= Age < 40 else 0
        AgeGroup_40_50 = 1 if 40 <= Age < 50 else 0
        AgeGroup_50_60 = 1 if 50 <= Age < 60 else 0
        AgeGroup_60_80 = 1 if 60 <= Age <= 80 else 0
        BMI_Category_Normal = 1 if 18.5 <= BMI < 25 else 0
        BMI_Category_Overweight = 1 if 25 <= BMI < 30 else 0
        BMI_Category_Obese = 1 if 30 <= BMI < 40 else 0
        BMI_Category_Severely_Obese = 1 if BMI >= 40 else 0

        input_data = {
            'Pregnancies': [Pregnancies],
            'Glucose': [Glucose],
            'BMI': [BMI],
            'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
            'Age': [Age],
            'Glucose_BMI': [Glucose_BMI],
            'Age_Pedigree': [Age_Pedigree],
            'AgeGroup_30-40': [AgeGroup_30_40],
            'AgeGroup_40-50': [AgeGroup_40_50],
            'AgeGroup_50-60': [AgeGroup_50_60],
            'AgeGroup_60-80': [AgeGroup_60_80],
            'BMI_Category_Normal': [BMI_Category_Normal],
            'BMI_Category_Overweight': [BMI_Category_Overweight],
            'BMI_Category_Obese': [BMI_Category_Obese],
            'BMI_Category_Severely Obese': [BMI_Category_Severely_Obese],
        }

        if st.sidebar.button("Predict"):
            result, probabilities = prediction(input_data)
            if result == 1:
                st.markdown(
                    f"""
                    <div style="padding:20px; background-color:#ffcccc; border-radius:5px;">
                        <h2 style="color:red; text-align:center;">
                            🚨 Prediction: Diabetic 🚨
                        </h2>
                        <p>Probability: {probabilities[1]:.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.snow()
            else:
                st.markdown(
                    f"""
                    <div style="padding:20px; background-color:#ccffcc; border-radius:5px;">
                        <h2 style="color:green; text-align:center;">
                            ✅ Prediction: Non-Diabetic ✅
                        </h2>
                        <p>Probability: {probabilities[0]:.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.balloons()

    with eda_tab:
        st.header("Exploratory Data Analysis")
        diabetes = pd.read_csv("diabetes.csv")
        st.write("### Dataset Overview")
        st.dataframe(diabetes.head())

        st.write("### Distribution of Features")
        selected_feature = st.selectbox("Select a feature to visualize:", diabetes.columns[:-1])
        fig, ax = plt.subplots()
        sns.histplot(diabetes[selected_feature], kde=True, ax=ax, color="blue")
        st.pyplot(fig)
        st.markdown(f"This graph shows the distribution of the {selected_feature} feature.")

        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(diabetes.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.write("### Additional Graphs")
        st.write("#### BMI vs Glucose by Outcome")
        fig, ax = plt.subplots()
        sns.scatterplot(data=diabetes, x="BMI", y="Glucose", hue="Outcome", ax=ax, alpha=0.8)
        ax.set_title("BMI vs Glucose")
        st.pyplot(fig)

        st.write("#### Age Distribution by Outcome")
        fig, ax = plt.subplots()
        sns.histplot(data=diabetes, x="Age", hue="Outcome", kde=True, element="step", bins=20, ax=ax)
        ax.set_title("Age Distribution by Outcome")
        st.pyplot(fig)

        st.write("#### Boxplot of BMI by Outcome")
        fig, ax = plt.subplots()
        sns.boxplot(data=diabetes, x="Outcome", y="BMI", palette="coolwarm", ax=ax)
        ax.set_title("Boxplot of BMI by Outcome")
        st.pyplot(fig)

    with feature_engineering_tab:
        st.header("Feature Engineering")
        st.write("### Derived Features")
        st.markdown(
            "* **Glucose_BMI**: A product of Glucose and BMI to capture interaction.\n"
            "* **Age_Pedigree**: Age multiplied by Diabetes Pedigree Function.\n"
            "* **Categorical Features**: Age groups and BMI categories."
        )

    with dataset_desc_tab:
        st.header("Dataset Description")
        st.write("### Descriptive Statistics")
        st.dataframe(diabetes.describe())

    with model_insights_tab:
        st.header("Model Insights")
        st.write("### Feature Importances")
        feature_importances = pd.DataFrame({
            'Feature': classifier.feature_names_in_,
            'Importance': classifier.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(feature_importances.set_index("Feature"))
        st.markdown("The bar chart shows the importance of each feature in the model.")

if __name__ == '__main__':
    main()
