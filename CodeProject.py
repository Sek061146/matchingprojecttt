import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import spacy
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from jupyter_dash import JupyterDash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import base64
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
sheet_id = "111buGxvuvwu1kHzWmkHOmdzSeV8xXgCH-ZoK4MZET0o"  # อยู่ในลิงก์ของ Google Sheet
sheet_name = "Job Description (project)"

url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name.replace(' ', '%20')}"
job = pd.read_csv(url)
job.head()

job['Age'] = job['Age'].astype('Int64')

job = job.drop(columns=['Unnamed: 14'])

job.shape

job.isnull().sum()

job.info()

    # Download NLTK data if you haven't already
try:
        nltk.data.find('corpora/stopwords')
except LookupError:
        nltk.download('stopwords')
try:
        nltk.data.find('tokenizers/punkt')
except LookupError:
        nltk.download('punkt')
try:
        nltk.data.find('corpora/wordnet')
except LookupError:
        nltk.download('wordnet')


nlp = spacy.load("en_core_web_sm")


# ✅ โหลดโมเดล SBERT
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# ✅ ข้อมูลตำแหน่งงาน
job_skill_data = {
    "Data Analyst": {
        "Years Experience": 2,
        "Education Level": "Bachelor",
        "Hard-Skills": [
            "Excel", "SQL", "Power BI", "Tableau", "Google Sheets",
            "Python", "Pandas", "Data Visualization", "Statistics", "Looker", "R"
        ],
        "Soft-Skills": [
            "Communication", "Critical Thinking", "Attention to Detail", "Business Understanding"
        ]
    },
    "Data Scientist": {
        "Years Experience": 3,
        "Education Level": "Master",
        "Hard-Skills": [
            "Python", "R", "Machine Learning", "Deep Learning", "Scikit-learn",
            "TensorFlow", "PyTorch", "SQL", "Statistics", "Data Cleaning", "Feature Engineering"
        ],
        "Soft-Skills": [
            "Problem Solving", "Curiosity", "Creativity", "Storytelling with Data", "Collaboration"
        ]
    },
    "Data Engineer": {
        "Years Experience": 5,
        "Education Level": "Bachelor",
        "Hard-Skills": [
            "Python", "SQL", "Spark", "Hadoop", "ETL", "Airflow", "AWS",
            "GCP", "BigQuery", "Snowflake", "Data Warehouse", "Data Pipeline Architecture"
        ],
        "Soft-Skills": [
            "System Thinking", "Communication", "Debugging Skills", "Collaboration", "Time Management"
        ]
    },
    "Business Intelligence": {
        "Years Experience": 4,
        "Education Level": "Bachelor",
        "Hard-Skills": [
            "SQL", "Power BI", "Tableau", "Excel", "Data Modeling", "DAX",
            "ETL Tools", "Database Design", "Dashboard Design", "Reporting Tools"
        ],
        "Soft-Skills": [
            "Business Acumen", "Presentation Skills", "Analytical Thinking",
            "Communication", "Stakeholder Management"
        ]
    }
}

edu_bonus_map = {'None': 0, 'Bachelor': 0, 'Master': 5, 'PhD': 10}

# ✅ เตรียม job profile เป็นข้อความสำหรับแต่ละตำแหน่ง
job_profiles = []
records = []
for position, data in job_skill_data.items():
    full_text = ", ".join(data["Hard-Skills"] + data["Soft-Skills"])
    job_profiles.append(full_text)
    records.append({
        "Position": position,
        "Years Experience": data["Years Experience"],
        "Education Level": data["Education Level"],
        "Profile Text": full_text
    })

job = pd.DataFrame(records)

# ✅ ข้อมูล Resume
resume_text = "Python, SQL, data visualization, good communication, TensorFlow, Machine Learning, 3 years experience, Master's Degree in Data Science"
education_level_input = "Master"

# ✅ ดึงปีประสบการณ์
def extract_years_experience(text):
    match = re.search(r'(\d+)\s*(?:years?|yrs?)', text.lower())
    return int(match.group(1)) if match else 0

resume_years = extract_years_experience(resume_text)
resume_edu_bonus = edu_bonus_map.get(education_level_input, 0)

# ✅ แปลงเป็น embedding ด้วย SBERT
resume_embed = model.encode([resume_text])[0]
job_embeds = model.encode(job["Profile Text"].tolist())

# ✅ ฟังก์ชันหา Skills ที่ยังขาด
def find_missing_skills(job_row, resume_text):
    resume_keywords = set([s.strip().lower() for s in resume_text.lower().replace("years experience", "").replace("master's degree", "").split(',')])
    job_hard = set(s.lower() for s in job_skill_data[job_row["Position"]]["Hard-Skills"])
    job_soft = set(s.lower() for s in job_skill_data[job_row["Position"]]["Soft-Skills"])
    missing_hard = [s for s in job_hard if s not in resume_keywords]
    missing_soft = [s for s in job_soft if s not in resume_keywords]
    return missing_hard, missing_soft

# ✅ คำนวณคะแนน + สร้างผลลัพธ์
similarities = cosine_similarity([resume_embed], job_embeds)[0]
final_scores = similarities * 100 + resume_edu_bonus

output = []
for i, row in job.iterrows():
    missing_hard, missing_soft = find_missing_skills(row, resume_text)
    output.append({
        "Position": row["Position"],
        "Match Score (%)": round(final_scores[i], 2),
        "Experience Required": row["Years Experience"],
        "Your Experience": resume_years,
        "Experience OK?": resume_years >= row["Years Experience"],
        "Missing Hard-Skills": ", ".join(missing_hard) if missing_hard else "✅ None",
        "Missing Soft-Skills": ", ".join(missing_soft) if missing_soft else "✅ None"
    })

# ✅ แสดงผลลัพธ์
df_result = pd.DataFrame(output).sort_values(by="Match Score (%)", ascending=False)
print(df_result)


# ✅ Load SBERT
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# ✅ Job skill data
job_skill_data = {
    "Data Analyst": {
        "Years Experience": 2,
        "Education Level": "Bachelor",
        "Hard-Skills": [
            "Excel", "SQL", "Power BI", "Tableau", "Google Sheets",
            "Python", "Pandas", "Data Visualization", "Statistics", "Looker", "R"
        ],
        "Soft-Skills": [
            "Communication", "Critical Thinking", "Attention to Detail", "Business Understanding"
        ]
    },
    "Data Scientist": {
        "Years Experience": 3,
        "Education Level": "Master",
        "Hard-Skills": [
            "Python", "R", "Machine Learning", "Deep Learning", "Scikit-learn",
            "TensorFlow", "PyTorch", "SQL", "Statistics", "Data Cleaning", "Feature Engineering"
        ],
        "Soft-Skills": [
            "Problem Solving", "Curiosity", "Creativity", "Storytelling with Data", "Collaboration"
        ]
    },
    "Data Engineer": {
        "Years Experience": 5,
        "Education Level": "Bachelor",
        "Hard-Skills": [
            "Python", "SQL", "Spark", "Hadoop", "ETL", "Airflow", "AWS",
            "GCP", "BigQuery", "Snowflake", "Data Warehouse", "Data Pipeline Architecture"
        ],
        "Soft-Skills": [
            "System Thinking", "Communication", "Debugging Skills", "Collaboration", "Time Management"
        ]
    },
    "Business Intelligence": {
        "Years Experience": 4,
        "Education Level": "Bachelor",
        "Hard-Skills": [
            "SQL", "Power BI", "Tableau", "Excel", "Data Modeling", "DAX",
            "ETL Tools", "Database Design", "Dashboard Design", "Reporting Tools"
        ],
        "Soft-Skills": [
            "Business Acumen", "Presentation Skills", "Analytical Thinking",
            "Communication", "Stakeholder Management"
        ]
    }
}

edu_bonus_map = {'None': 0, 'Bachelor': 0, 'Master': 5, 'PhD': 10}

# ✅ เตรียม job profiles
job_profiles = []
records = []
for position, data in job_skill_data.items():
    full_text = ", ".join(data["Hard-Skills"] + data["Soft-Skills"])
    job_profiles.append(full_text)
    records.append({
        "Position": position,
        "Years Experience": data["Years Experience"],
        "Education Level": data["Education Level"],
        "Profile Text": full_text
    })

job_df = pd.DataFrame(records)

# ✅ Layout
app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    html.H2("\U0001F4C4 Upload Resume (PDF)"),
    dcc.Upload(
        id='upload-pdf',
        children=html.Div(['Drag & Drop or ', html.A('Select PDF File')]),
        style={
            'width': '100%', 'height': '120px', 'lineHeight': '60px',
            'borderWidth': '2px', 'borderStyle': 'dashed',
            'borderRadius': '10px', 'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),
    html.Div([
        html.Label("\U0001F393 Select your education level:"),
        dcc.Dropdown(
            id='edu-level',
            options=[{'label': k, 'value': k} for k in edu_bonus_map.keys()],
            value='Master',
            clearable=False
        )
    ], style={'margin': '10px 0'}),
    html.Hr(),
    html.H4("\U0001F4DC Analysis Results"),
    html.Pre(id='output-text', style={'whiteSpace': 'pre-wrap', 'backgroundColor': '#f8f9fa', 'padding': '20px'})
], fluid=True)

# ✅ Utils
def extract_years_experience(text):
    match = re.search(r'(\d+)\s*(?:years?|yrs?)', text.lower())
    return int(match.group(1)) if match else 0

def find_missing_skills(job_row, resume_text):
    resume_keywords = set([s.strip().lower() for s in resume_text.lower().replace("years experience", "").replace("master's degree", "").split(',')])
    job_hard = set(s.lower() for s in job_skill_data[job_row["Position"]]["Hard-Skills"])
    job_soft = set(s.lower() for s in job_skill_data[job_row["Position"]]["Soft-Skills"])
    missing_hard = [s for s in job_hard if s not in resume_keywords]
    missing_soft = [s for s in job_soft if s not in resume_keywords]
    return missing_hard, missing_soft

def analyze_resume(text, education_level_input):
    resume_years = extract_years_experience(text)
    resume_edu_bonus = edu_bonus_map.get(education_level_input, 0)
    resume_embed = model.encode([text[:3000]])[0]  # ✨ Limit length
    job_embeds = model.encode(job_df["Profile Text"].tolist())
    similarities = cosine_similarity([resume_embed], job_embeds)[0]
    final_scores = similarities * 100 + resume_edu_bonus

    output = []
    for i, row in job_df.iterrows():
        missing_hard, missing_soft = find_missing_skills(row, text)
        output.append({
            "Position": row["Position"],
            "Match Score (%)": round(final_scores[i], 2),
            "Experience Required": row["Years Experience"],
            "Your Experience": resume_years,
            "Experience OK?": resume_years >= row["Years Experience"],
            "Missing Hard-Skills": ", ".join(missing_hard) if missing_hard else "\u2705 None",
            "Missing Soft-Skills": ", ".join(missing_soft) if missing_soft else "\u2705 None"
        })

    df_result = pd.DataFrame(output).sort_values(by="Match Score (%)", ascending=False)
    return df_result.to_string(index=False)

# ✅ Callback
@app.callback(
    Output('output-text', 'children'),
    Input('upload-pdf', 'contents'),
    State('edu-level', 'value')
)
def extract_and_analyze(contents, edu_level):
    if contents is None:
        return "\U0001F4CE \u0e01\u0e23\u0e38\u0e13\u0e32\u0e2d\u0e31\u0e1b\u0e42\u0e2b\u0e25\u0e14\u0e44\u0e1f\u0e25\u0e4c PDF \u0e01\u0e48\u0e2d\u0e19"
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        text = ""
        with fitz.open(stream=decoded, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return analyze_resume(text, education_level_input=edu_level)
    except Exception as e:
        return f"\u274c \u0e40\u0e01\u0e34\u0e14\u0e02\u0e49\u0e2d\u0e1c\u0e25\u0e1e\u0e25\u0e32\u0e14: {str(e)}"

app.run_server(debug=True, host="0.0.0.0", port=8000)

