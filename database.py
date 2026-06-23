"""
Run this script once to initialize the database.
The app.py also auto-initializes on startup.
"""
import sqlite3, hashlib

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

conn = sqlite3.connect("placement.db")
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    name TEXT, email TEXT, prediction TEXT, probability REAL,
    degree_p REAL, skills_rating INTEGER, dsa_rating INTEGER,
    communication_rating INTEGER, projects_count INTEGER, internships_count INTEGER,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)""")

c.execute("""CREATE TABLE IF NOT EXISTS students(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, password TEXT NOT NULL,
    roll_no TEXT, branch TEXT, year TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)""")

c.execute("""CREATE TABLE IF NOT EXISTS admins(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL, password TEXT NOT NULL
)""")

c.execute("SELECT * FROM admins WHERE username='admin'")
if not c.fetchone():
    c.execute("INSERT INTO admins(username,password) VALUES(?,?)",
              ('admin', hash_password('admin123')))

conn.commit()
conn.close()
print("✅ Database initialized successfully.")
print("Admin credentials: admin / admin123")
