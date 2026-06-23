from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import pickle
import numpy as np
import sqlite3
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
import hashlib
import os

app = Flask(__name__)
app.secret_key = 'placement_secret_key_2024_secure'

# ─── Load ML Models ────────────────────────────────────────────────────────────
model    = pickle.load(open("models/model.pkl",    "rb"))
scaler   = pickle.load(open("models/scaler.pkl",   "rb"))
encoders = pickle.load(open("models/encoders.pkl", "rb"))

# ─── Helpers ───────────────────────────────────────────────────────────────────
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def get_db():
    conn = sqlite3.connect("placement.db")
    conn.row_factory = sqlite3.Row
    return conn

# ─── DB Init ───────────────────────────────────────────────────────────────────
def init_db():
    conn = get_db()
    c = conn.cursor()

    # predictions table
    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        name TEXT,
        email TEXT,
        prediction TEXT,
        probability REAL,
        degree_p REAL,
        skills_rating INTEGER,
        dsa_rating INTEGER,
        communication_rating INTEGER,
        projects_count INTEGER,
        internships_count INTEGER,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # students table
    c.execute("""
    CREATE TABLE IF NOT EXISTS students(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        roll_no TEXT,
        branch TEXT,
        year TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # admins table
    c.execute("""
    CREATE TABLE IF NOT EXISTS admins(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )""")

    # Default admin: admin / admin123
    c.execute("SELECT * FROM admins WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO admins(username,password) VALUES(?,?)",
                  ('admin', hash_password('admin123')))

    # Demo student
    c.execute("SELECT * FROM students WHERE email='demo@student.com'")
    if not c.fetchone():
        c.execute("""INSERT INTO students(name,email,password,roll_no,branch,year)
                     VALUES(?,?,?,?,?,?)""",
                  ('Demo Student','demo@student.com', hash_password('demo123'),
                   'CS2021001','Computer Science','Final Year'))

    conn.commit()
    conn.close()

init_db()

# ═══════════════════════════════════════════════════════════
#  LANDING / HOME
# ═══════════════════════════════════════════════════════════
@app.route('/')
def landing():
    return render_template('landing.html')

# ═══════════════════════════════════════════════════════════
#  STUDENT AUTH
# ═══════════════════════════════════════════════════════════
@app.route('/student/login', methods=['GET','POST'])
def student_login():
    if request.method == 'POST':
        email = request.form['email'].strip()
        password = hash_password(request.form['password'])
        conn = get_db()
        student = conn.execute("SELECT * FROM students WHERE email=? AND password=?",
                               (email, password)).fetchone()
        conn.close()
        if student:
            session['student_id']   = student['id']
            session['student_name'] = student['name']
            session['student_email']= student['email']
            session['role']         = 'student'
            flash('Welcome back, ' + student['name'] + '!', 'success')
            return redirect(url_for('student_dashboard'))
        flash('Invalid email or password.', 'error')
    return render_template('student_login.html')

@app.route('/student/register', methods=['GET','POST'])
def student_register():
    if request.method == 'POST':
        name     = request.form['name'].strip()
        email    = request.form['email'].strip()
        password = hash_password(request.form['password'])
        roll_no  = request.form.get('roll_no','').strip()
        branch   = request.form.get('branch','').strip()
        year     = request.form.get('year','').strip()
        try:
            conn = get_db()
            conn.execute("""INSERT INTO students(name,email,password,roll_no,branch,year)
                            VALUES(?,?,?,?,?,?)""",
                         (name,email,password,roll_no,branch,year))
            conn.commit()
            conn.close()
            flash('Account created! Please login.', 'success')
            return redirect(url_for('student_login'))
        except sqlite3.IntegrityError:
            flash('Email already registered.', 'error')
    return render_template('student_register.html')

@app.route('/student/logout')
def student_logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('student_login'))

# ═══════════════════════════════════════════════════════════
#  STUDENT DASHBOARD
# ═══════════════════════════════════════════════════════════
@app.route('/student/dashboard')
def student_dashboard():
    if session.get('role') != 'student':
        return redirect(url_for('student_login'))
    conn = get_db()
    history = conn.execute("""SELECT * FROM predictions WHERE student_id=?
                               ORDER BY date DESC""",
                           (session['student_id'],)).fetchall()
    best = conn.execute("""SELECT MAX(probability) FROM predictions WHERE student_id=?""",
                        (session['student_id'],)).fetchone()[0] or 0
    total = len(history)
    placed = sum(1 for r in history if 'LIKELY' in r['prediction'])
    conn.close()
    return render_template('student_dashboard.html',
                           history=history, best=best, total=total, placed=placed)

# ═══════════════════════════════════════════════════════════
#  PREDICTION FORM (Student)
# ═══════════════════════════════════════════════════════════
@app.route('/predict/form')
def predict_form():
    if session.get('role') != 'student':
        return redirect(url_for('student_login'))
    return render_template('index.html')

# ═══════════════════════════════════════════════════════════
#  PREDICT
# ═══════════════════════════════════════════════════════════
@app.route('/predict', methods=['POST'])
def predict():
    if session.get('role') != 'student':
        return redirect(url_for('student_login'))

    name  = request.form.get('name', session.get('student_name',''))
    email = session.get('student_email','')

    # Skill Inputs
    skills_rating        = int(request.form.get('skills_rating') or 5)
    dsa_rating           = int(request.form.get('dsa_rating') or 5)
    communication_rating = int(request.form.get('communication_rating') or 5)
    projects_count       = int(request.form.get('projects_count') or 0)
    internships_count    = int(request.form.get('internships_count') or 0)
    certifications_count = int(request.form.get('certifications_count') or 0)
    hackathon_count      = int(request.form.get('hackathon_count') or 0)

    # Numerical
    ssc_p    = float(request.form.get('ssc_p') or 0)
    hsc_p    = float(request.form.get('hsc_p') or 0)
    degree_p = float(request.form.get('degree_p') or 0)
    etest_p  = float(request.form.get('etest_p') or 0)
    mba_p    = float(request.form.get('mba_p') or 0)

    # Encode categoricals
    gender         = encoders['gender'].transform([request.form.get('gender')])[0]
    ssc_b          = encoders['ssc_b'].transform([request.form.get('ssc_b')])[0]
    hsc_b          = encoders['hsc_b'].transform([request.form.get('hsc_b')])[0]
    hsc_s          = encoders['hsc_s'].transform([request.form.get('hsc_s')])[0]
    degree_t       = encoders['degree_t'].transform([request.form.get('degree_t')])[0]
    workex         = encoders['workex'].transform([request.form.get('workex')])[0]
    spec_input     = request.form.get('specialisation')
    specialisation = encoders['specialisation'].transform([spec_input])[0] if spec_input else 0

    features = np.array([[gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s,
                          degree_p, degree_t, workex, etest_p, specialisation, mba_p]])
    features = scaler.transform(features)

    prediction       = model.predict(features)[0]
    raw_probability  = model.predict_proba(features)[0][1] * 100

    skill_score = (skills_rating*2 + dsa_rating*2 + communication_rating*1.5
                   + projects_count*3 + internships_count*5
                   + certifications_count*2 + hackathon_count*2)

    probability = 0.7 * raw_probability + 0.3 * skill_score
    probability = max(15, min(probability, 95))
    probability = round(probability, 2)

    # Result label
    if probability >= 70:
        result       = "LIKELY TO BE PLACED ✅"
        careers      = ["Software Engineer","Data Analyst","AI/ML Engineer","Web Developer"]
        suggestions  = ["Continue building projects","Strengthen DSA skills","Learn SQL and Git","Apply for internships"]
    else:
        result       = "LOW PLACEMENT CHANCES ❌"
        careers      = ["Python Developer","Data Analyst","Web Developer","Technical Support Engineer"]
        suggestions  = ["Improve Aptitude Skills","Build More Projects","Gain Internship Experience","Strengthen DSA"]

    if probability >= 80:
        chance_level   = "🟢 High Placement Chance"
        progress_color = "#28a745"
        advice = ["Excellent profile overall.","Continue building projects.",
                  "Strengthen DSA and problem solving.","Apply for internships."]
    elif probability >= 50:
        chance_level   = "🟡 Moderate Placement Chance"
        progress_color = "#ffc107"
        advice = ["You have decent chances.","Improve communication skills.",
                  "Build more projects.","Gain internship experience."]
    else:
        chance_level   = "🔴 Low Placement Chance"
        progress_color = "#dc3545"
        advice = ["Focus on improving academics.","Practice aptitude regularly.",
                  "Work on DSA and development skills.","Participate in hackathons."]

    if probability >= 85:
        companies = ["Microsoft","Google","Amazon","Adobe","NVIDIA","Goldman Sachs"]
    elif probability >= 70:
        companies = ["TCS Digital","Accenture","Infosys","Capgemini","Cognizant","Deloitte"]
    elif probability >= 50:
        companies = ["Wipro","Tech Mahindra","LTIMindtree","HCL Technologies","Persistent Systems"]
    else:
        companies = ["Focus on Internships","Build Strong Projects","Participate in Hackathons",
                     "Strengthen DSA","Improve Communication Skills"]

    strengths = []
    if degree_p >= 75:          strengths.append("Strong Academic Profile")
    if skills_rating >= 7:      strengths.append("Good Technical Skills")
    if projects_count >= 3:     strengths.append("Excellent Project Portfolio")
    if internships_count >= 1:  strengths.append("Internship Experience")
    if workex == 1:             strengths.append("Prior Work Experience")
    if hackathon_count >= 2:    strengths.append("Hackathon Participant")
    if not strengths:           strengths.append("Keep building your profile to unlock strengths.")

    weaknesses = []
    if dsa_rating < 7:              weaknesses.append("Continue DSA Practice")
    if communication_rating < 7:    weaknesses.append("Improve Communication Skills")
    if projects_count < 3:          weaknesses.append("Build More Projects")
    if internships_count == 0:      weaknesses.append("Gain Internship Experience")
    if certifications_count < 2:    weaknesses.append("Complete More Certifications")
    if not weaknesses:              weaknesses.append("Great profile! Maintain consistency.")

    technical_skill_percent  = skills_rating * 10
    dsa_percent              = dsa_rating * 10
    communication_percent    = communication_rating * 10
    project_percent          = min(projects_count * 20, 100)

    note = ("This prediction is based on historical placement data and skill assessment. "
            "It should be interpreted as an estimate rather than a guarantee.")

    # Save
    conn = get_db()
    conn.execute("""INSERT INTO predictions
                    (student_id,name,email,prediction,probability,degree_p,
                     skills_rating,dsa_rating,communication_rating,projects_count,internships_count)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                 (session['student_id'], name, email, result, probability, degree_p,
                  skills_rating, dsa_rating, communication_rating, projects_count, internships_count))
    conn.commit()
    pred_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    # Store for PDF
    session['latest_report'] = {
        "name": name, "prediction": result, "probability": probability,
        "strengths": strengths, "weaknesses": weaknesses, "careers": careers,
        "companies": companies, "suggestions": suggestions,
        "technical_skill_percent": technical_skill_percent,
        "dsa_percent": dsa_percent,
        "communication_percent": communication_percent,
        "project_percent": project_percent
    }

    return render_template('result.html',
        prediction=result, probability=probability, careers=careers,
        suggestions=suggestions, chance_level=chance_level,
        progress_color=progress_color, advice=advice, note=note,
        strengths=strengths, weaknesses=weaknesses,
        technical_skill_percent=technical_skill_percent,
        dsa_percent=dsa_percent, communication_percent=communication_percent,
        project_percent=project_percent, companies=companies)

# ═══════════════════════════════════════════════════════════
#  PDF DOWNLOAD
# ═══════════════════════════════════════════════════════════
@app.route('/download_report')
def download_report():
    if session.get('role') != 'student':
        return redirect(url_for('student_login'))
    lr = session.get('latest_report', {})
    if not lr:
        flash('No report found. Please predict first.', 'error')
        return redirect(url_for('predict_form'))

    report_date = datetime.now().strftime("%d-%m-%Y %H:%M")
    filename = "Placement_Report.pdf"
    doc = SimpleDocTemplate(filename, rightMargin=inch*0.75, leftMargin=inch*0.75,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    elements = []

    # Header
    header_style = ParagraphStyle('header', parent=styles['Title'],
                                  textColor=colors.HexColor('#1e40af'), spaceAfter=6)
    elements.append(Paragraph("AI PLACEMENT PREDICTION SYSTEM", header_style))
    elements.append(Paragraph("Student Analysis Report", styles['Heading2']))
    elements.append(Spacer(1, 12))

    # Info table
    data = [
        ["Candidate Name", lr.get('name','N/A')],
        ["Report Generated", report_date],
        ["Prediction", lr.get('prediction','N/A')],
        ["Placement Probability", f"{lr.get('probability','N/A')}%"]
    ]
    t = Table(data, colWidths=[2.5*inch, 4*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#1e40af')),
        ('TEXTCOLOR', (0,0), (0,-1), colors.white),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (1,0), (1,-1), [colors.HexColor('#eff6ff'), colors.white]),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))

    def section(title, items):
        elements.append(Paragraph(title, styles['Heading2']))
        for item in items:
            elements.append(Paragraph(f"  • {item}", styles['Normal']))
        elements.append(Spacer(1, 14))

    section("✅ Strengths",           lr.get('strengths',[]))
    section("⚡ Areas to Improve",    lr.get('weaknesses',[]))
    section("🚀 Recommended Careers", lr.get('careers',[]))
    section("🏢 Target Companies",    lr.get('companies',[]))
    section("💡 Suggestions",         lr.get('suggestions',[]))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        "<font color='grey'>Generated by AI Placement Prediction System | Confidential</font>",
        styles['Italic']))

    doc.build(elements)
    return send_file(filename, as_attachment=True)

# ═══════════════════════════════════════════════════════════
#  ADMIN AUTH
# ═══════════════════════════════════════════════════════════
@app.route('/admin/login', methods=['GET','POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = hash_password(request.form['password'])
        conn = get_db()
        admin = conn.execute("SELECT * FROM admins WHERE username=? AND password=?",
                             (username, password)).fetchone()
        conn.close()
        if admin:
            session['admin_id']   = admin['id']
            session['admin_name'] = admin['username']
            session['role']       = 'admin'
            flash('Welcome, Admin!', 'success')
            return redirect(url_for('admin_dashboard'))
        flash('Invalid credentials.', 'error')
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.clear()
    flash('Admin logged out.', 'info')
    return redirect(url_for('admin_login'))

# ═══════════════════════════════════════════════════════════
#  ADMIN DASHBOARD
# ═══════════════════════════════════════════════════════════
@app.route('/admin/dashboard')
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    conn = get_db()
    rows  = conn.execute("SELECT * FROM predictions ORDER BY date DESC").fetchall()
    students = conn.execute("SELECT * FROM students ORDER BY created_at DESC").fetchall()
    conn.close()

    total_predictions = len(rows)
    total_placed      = sum(1 for r in rows if 'LIKELY' in r['prediction'])
    total_not_placed  = total_predictions - total_placed
    total_students    = len(students)

    avg_prob = round(sum(r['probability'] for r in rows) / total_predictions, 2) if rows else 0
    placement_rate = round((total_placed / total_predictions)*100, 2) if rows else 0

    top_candidate = max(rows, key=lambda r: r['probability'], default=None)
    trend_labels  = [str(r['id']) for r in rows[-20:]]
    trend_values  = [r['probability'] for r in rows[-20:]]

    # Degree-wise avg
    degree_buckets = {'<60': 0, '60-70': 0, '70-80': 0, '80+': 0}
    for r in rows:
        dp = r['degree_p'] or 0
        if dp < 60:   degree_buckets['<60'] += 1
        elif dp < 70: degree_buckets['60-70'] += 1
        elif dp < 80: degree_buckets['70-80'] += 1
        else:         degree_buckets['80+'] += 1

    return render_template('admin.html',
        rows=rows, students=students,
        total_predictions=total_predictions, total_placed=total_placed,
        total_not_placed=total_not_placed, total_students=total_students,
        average_probability=avg_prob, placement_rate=placement_rate,
        top_candidate=top_candidate,
        trend_labels=trend_labels, trend_values=trend_values,
        degree_buckets=degree_buckets)

# ═══════════════════════════════════════════════════════════
#  ADMIN – MANAGE STUDENTS
# ═══════════════════════════════════════════════════════════
@app.route('/admin/students')
def admin_students():
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    conn = get_db()
    students = conn.execute("SELECT * FROM students ORDER BY created_at DESC").fetchall()
    conn.close()
    return render_template('admin_students.html', students=students)

@app.route('/admin/delete_student/<int:sid>')
def delete_student(sid):
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    conn = get_db()
    conn.execute("DELETE FROM students WHERE id=?", (sid,))
    conn.execute("DELETE FROM predictions WHERE student_id=?", (sid,))
    conn.commit()
    conn.close()
    flash('Student deleted.', 'success')
    return redirect(url_for('admin_students'))

@app.route('/admin/delete_prediction/<int:pid>')
def delete_prediction(pid):
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    conn = get_db()
    conn.execute("DELETE FROM predictions WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    flash('Prediction deleted.', 'success')
    return redirect(url_for('admin_dashboard'))

# ═══════════════════════════════════════════════════════════
#  HISTORY (student-specific)
# ═══════════════════════════════════════════════════════════
@app.route('/history')
def history():
    if session.get('role') != 'student':
        return redirect(url_for('student_login'))
    conn = get_db()
    rows = conn.execute("""SELECT * FROM predictions WHERE student_id=?
                           ORDER BY date DESC""", (session['student_id'],)).fetchall()
    conn.close()
    return render_template('history.html', rows=rows)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)