# 🎓 PlaceAI – Student Placement Prediction System

A full-featured ML-powered web application that predicts student placement probability using academic data and skill assessment.

---

## 🚀 Features

### Student Portal
- **Register / Login** – Secure student accounts with hashed passwords
- **Placement Prediction** – Enter academics + skills → get instant AI prediction
- **Personal Dashboard** – View your history, best score, success rate
- **Prediction History** – All past predictions with timestamps
- **PDF Report Download** – Professional PDF with strengths, companies, career advice

### Admin Panel
- **Secure Admin Login** – Separate admin authentication
- **Analytics Dashboard** – Total students, predictions, placement rate, charts
- **Probability Trend Line** – Visual trend of recent predictions
- **Placed vs Not Placed Pie Chart** – Quick visual split
- **Academic Performance Bar Chart** – Degree % distribution
- **Manage Students** – View and delete student accounts
- **Manage Predictions** – View and delete individual predictions
- **Top Candidate Card** – Highlights the best-performing student

### ML Backend
- **Random Forest Classifier** – Trained on real placement dataset
- **Hybrid Scoring** – Combines ML probability (70%) + skill score (30%)
- **Feature Engineering** – 12 academic + 7 skill features
- **Label Encoding + Standard Scaling** – Preprocessing pipeline

---

## 🔐 Default Credentials

| Role    | Username/Email        | Password   |
|---------|-----------------------|------------|
| Admin   | admin                 | admin123   |
| Student | demo@student.com      | demo123    |

---

## 📦 Setup

```bash
pip install flask reportlab scikit-learn numpy
python database.py   # Initialize DB (optional – app auto-inits)
python app.py        # Start server → http://localhost:5000
```

---

## 📁 Project Structure

```
Placement-Prediction-System/
├── app.py              # Main Flask app with all routes
├── database.py         # DB initialization script
├── models/
│   ├── model.pkl       # Trained Random Forest model
│   ├── scaler.pkl      # Standard Scaler
│   └── encoders.pkl    # Label Encoders
├── templates/
│   ├── landing.html    # Home page
│   ├── student_login.html
│   ├── student_register.html
│   ├── student_dashboard.html
│   ├── index.html      # Prediction form
│   ├── result.html     # Prediction result
│   ├── history.html    # Student prediction history
│   ├── admin_login.html
│   ├── admin.html      # Admin dashboard
│   └── admin_students.html
├── static/
│   └── style.css       # Complete responsive CSS
└── data/
    └── Placement_Data_Full_Class.csv
```

---

## 🧠 ML Model Details

- **Algorithm:** Random Forest Classifier
- **Features:** SSC%, HSC%, Degree%, E-Test%, MBA%, Gender, Boards, Stream, Degree Type, Work Experience, Specialisation
- **Skill Features (custom):** Technical Skills, DSA, Communication, Projects, Internships, Certifications, Hackathons
- **Hybrid Score:** `0.7 × ML_Probability + 0.3 × Skill_Score`
- **Output Tiers:** High (≥80%), Moderate (50–79%), Low (<50%)

---

## 📊 Admin Analytics

- 📈 Probability Trend Line Chart
- 🍩 Placed vs Not Placed Doughnut Chart
- 📚 Academic Performance Distribution Bar Chart
- 🏆 Top Candidate Highlight
- 📋 Full Prediction Table with Delete
- 👥 Student Management Table with Delete
