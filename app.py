import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("./Datasets/College Acceptance.csv", encoding='latin-1')

# Dropping unnecessary columns
df = df.drop(["Student ID", "College Name"], axis=1)

# Remove '%' symbol and convert "Chance of Acceptance" to numeric values
df['Chance of Acceptance'] = df['Chance of Acceptance'].str.rstrip('%').astype(float) / 100

# Apply sigmoid transformation to the target variable
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

df['Chance of Acceptance'] = df['Chance of Acceptance'].apply(sigmoid)

# Encoding categorical columns
le_course = LabelEncoder()
df['Course'] = le_course.fit_transform(df['Course'])

# Split the data into features and target variable
X = df.drop('Chance of Acceptance', axis=1)
y = df['Chance of Acceptance']

# Scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

def predict_acceptance_chance(high_school_gpa, english_score, exam_score, study_gap, num_backlog, course):
    english_scores = ['TOEFL', 'IELTS Academic', 'PTE', 'Duolingo', 'Cambridge']
    english_score_idx = english_scores.index(english_score)
    exam_score_idx = english_score_idx + 1
    course_encoded = le_course.transform([course])[0]
    
    input_data = np.zeros((1, len(X.columns)))
    input_data[:, 0] = high_school_gpa
    input_data[:, exam_score_idx] = exam_score
    input_data[:, 6] = study_gap
    input_data[:, 7] = num_backlog
    input_data[:, 8] = course_encoded
    
    input_data_scaled = scaler.transform(input_data)
    
    return model.predict(input_data_scaled)[0]

def main():
    st.title("College Acceptance Chance Prediction")
    st.markdown(
        """
        <style>
        .title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #0066CC;
        }
        .input-label {
            font-size: 18px;
            font-weight: bold;
            color: #333333;
        }
        .prediction-label {
            font-size: 24px;
            font-weight: bold;
            color: #0066CC;
            margin-top: 30px;
        }
        .prediction-value {
            font-size: 36px;
            font-weight: bold;
            color: #00A000;
        }
        .predict-button {
            margin-top: 30px;
            background-color: #0066CC;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }
        </style>
        """
        , unsafe_allow_html=True
    )

    st.markdown('<p class="title">Enter Student Information</p>', unsafe_allow_html=True)

    high_school_gpa = st.number_input("High School GPA", min_value=0.0, max_value=5.0, step=0.01)
    study_gap = st.number_input("Study Gap (in months)", min_value=0, max_value=None, step=1)
    num_backlog = st.number_input("Number of Backlog", min_value=0, max_value=None, step=1)

    english_scores = ['TOEFL', 'IELTS Academic', 'PTE', 'Duolingo', 'Cambridge']
    english_score = st.selectbox("English Language Score", english_scores)

    exam_score = st.number_input(f"Enter {english_score} Score", min_value=0, max_value=None, step=1)

    available_courses = sorted(le_course.inverse_transform(df['Course'].unique()))
    course = st.selectbox("Choose a Course", available_courses)

    if st.button("Predict Chance of Acceptance", key="predict"):
        chance_of_acceptance = predict_acceptance_chance(high_school_gpa, english_score, exam_score, study_gap, num_backlog, course)
        st.markdown('<p class="prediction-label">Predicted Chance of Acceptance</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="prediction-value">{chance_of_acceptance:.2%}</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
