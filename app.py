import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("./Datasets/College Acceptance.csv", encoding='latin-1')

# Encode the categorical variable
le = LabelEncoder()
df['Course'] = le.fit_transform(df['Course'])
le_college = LabelEncoder()
df['College Name'] = le_college.fit_transform(df['College Name'])

# Split the data into features and target variable
X = df.drop('Chance of Acceptance', axis=1)
y = df['Chance of Acceptance']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Define the available courses
available_courses = le.inverse_transform(df['Course'].unique())

# Create a Streamlit app
def main():
    # Set the app title
    st.title("College Admission Predictor")
    
    # Get user input
    st.subheader("Enter Your Details")
    
    high_school_gpa = st.number_input("High School GPA", min_value=0.0, step=0.1)
    
    english_exam_choice = st.selectbox("English Language Exam", ("TOEFL", "IELTS Academic", "PTE", "Duolingo", "Cambridge"))
    english_exam_score = st.number_input(f"{english_exam_choice} Score", min_value=0.0, step=0.1)
    
    study_gap = st.number_input("Study Gap (in years)", min_value=0, step=1)
    
    num_backlogs = st.number_input("Number of Backlogs", min_value=0, step=1)
    
    course_choice = st.selectbox("Course", available_courses)
    
    # Prepare the input data
    input_data = {
        'High School GPA': [high_school_gpa],
        'Course': [course_choice],
        'Study Gap': [study_gap],
        'Number of Backlog': [num_backlogs]
    }
    
    if english_exam_choice == "TOEFL":
        input_data['TOEFL'] = [english_exam_score]
    elif english_exam_choice == "IELTS Academic":
        input_data['IELTS Academic'] = [english_exam_score]
    elif english_exam_choice == "PTE":
        input_data['PTE'] = [english_exam_score]
    elif english_exam_choice == "Duolingo":
        input_data['Duolingo'] = [english_exam_score]
    elif english_exam_choice == "Cambridge":
        input_data['Cambridge'] = [english_exam_score]
    
    input_df = pd.DataFrame(input_data)
    
    # Predict chances of acceptance for the input data
    input_df_encoded = pd.get_dummies(input_df, columns=['Course'])
    
    # Make sure the input DataFrame has the same columns as the training DataFrame
    missing_columns = set(X.columns) - set(input_df_encoded.columns)
    for column in missing_columns:
        input_df_encoded[column] = 0
    
    # Reorder the columns to match the training DataFrame
    input_df_encoded = input_df_encoded[X.columns]
    
    prediction = model.predict(input_df_encoded)
    
    # Display the prediction result
    st.subheader("Prediction Result")
    for college_index, chance in zip(df['College Name'].unique(), prediction):
        college_name = le_college.inverse_transform([college_index])[0]
        chance = f"{chance:.2f}%"
        st.write(f"- {college_name}: {chance}")
    
# Run the app
if __name__ == "__main__":
    main()
