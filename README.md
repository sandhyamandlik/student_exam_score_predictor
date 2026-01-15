# 🎓 Student Exam Score Predictor

This project is a machine learning-based web application that predicts a student's exam score based on various academic, socio-economic, and lifestyle factors. It is built using Python, Flask, Scikit-learn, and a simple HTML/CSS frontend.

# Features

- Predicts exam scores using a trained Random Forest Regressor model.
- Easy-to-use interactive web form for input.
- Uses a diverse dataset with 20+ influencing factors.
- Real-time prediction via REST API.


# Technologies Used
- Python, Flask
- Scikit-learn, Pandas, NumPy
- HTML, CSS (Dark Theme UI)
- Flask-CORS for frontend-backend integration


# Dataset

The dataset `StudentPerformanceFactors.csv` contains records of student demographics, academic resources, personal habits, and their corresponding `Exam_Score`. It includes features like:

- Hours Studied
- Attendance
- Parental Involvement
- Access to Resources
- Motivation Level
- Sleep Hours
- Gender, Family Income, and more...

# How to Run Locally 

1. git clone https://github.com/your-username/student-score-predictor.git
2. cd student-score-predictor
3. pip install flask pandas scikit-learn joblib flask-cors
4. python retrain_model.py
5. python student_score_predictor.py

