# Import necessary libraries
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of students
num_students = 100

# Generate random data
attendance = np.random.uniform(60, 100, num_students)  # Attendance percentage
quiz_scores = np.random.uniform(5, 10, num_students)  # Quiz scores (out of 10)
homework_completion = np.random.uniform(50, 100, num_students)  # Homework completion

# Calculate final grades with random noise
final_grades = (
    0.3 * attendance + 
    0.4 * quiz_scores * 10 + 
    0.3 * homework_completion + 
    np.random.normal(0, 5, num_students)
)

# Clip grades to 0-100 and assign letter grades
final_grades = np.clip(final_grades, 0, 100)
letter_grades = pd.cut(
    final_grades, 
    bins=[0, 60, 70, 80, 90, 100], 
    labels=["F", "D", "C", "B", "A"]
)

# Create a DataFrame
student_data = pd.DataFrame({
    "Student ID": range(1, num_students + 1),
    "Attendance (%)": attendance.round(2),
    "Quiz Scores (Avg)": quiz_scores.round(2),
    "Homework Completion (%)": homework_completion.round(2),
    "Final Grade (%)": final_grades.round(2),
    "Letter Grade": letter_grades
})

# Save to CSV and print first few rows
student_data.to_csv("student_performance.csv", index=False)
print("Dataset created and saved as 'student_performance.csv'")
print(student_data.head())

# Import necessary libraries for modeling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Prepare data for machine learning
X = student_data[["Attendance (%)", "Quiz Scores (Avg)", "Homework Completion (%)"]]
y = student_data["Letter Grade"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
