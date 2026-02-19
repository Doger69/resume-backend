from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import PyPDF2
import pdfplumber
import re
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

# -------------------- SKILL DEFINITIONS --------------------

SKILL_PATTERNS = {
    "Python": [r"\bpython\b"],
    "FastAPI": [r"\bfastapi\b"],
    "Django": [r"\bdjango\b"],
    "Flask": [r"\bflask\b"],
    "REST APIs": [r"\brest\b", r"\bapi\b", r"\bapis\b", r"\brestful\b"],
    "SQL Databases": [r"\bsql\b", r"\bmysql\b", r"\bpostgresql\b", r"\bpostgres\b"],
    "NoSQL Databases": [r"\bnosql\b", r"\bmongodb\b", r"\bdynamodb\b"],
    "Docker": [r"\bdocker\b"],
    "Kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "Git": [r"\bgit\b", r"\bgithub\b", r"\bgitlab\b"],
    "CI/CD Pipelines": [r"\bci\s*/\s*cd\b", r"\bcicd\b", r"\bjenkins\b", r"\bgithub\s+actions\b", r"\bpipeline\b"],
    "Cloud Platforms": [r"\baws\b", r"\bazure\b", r"\bgcp\b", r"\bcloud\b"],
    "Machine Learning": [r"\bmachine\s+learning\b", r"\bml\b", r"\bdeep\s+learning\b"],
    "NLP": [r"\bnlp\b", r"\bnatural\s+language\s+processing\b"],
    "Data Science": [r"\bdata\s+science\b", r"\bpandas\b", r"\bnumpy\b"],
    "TensorFlow / PyTorch": [r"\btensorflow\b", r"\bpytorch\b", r"\bkeras\b"],
    "Scalability": [r"\bscalabilit\w+\b", r"\bscalable\b"],
    "Performance Optimization": [r"\bperformance\b", r"\boptimization\b"],
    "Security": [r"\bsecurity\b", r"\bauth\w*\b", r"\boauth\b"],
    "Microservices": [r"\bmicroservice\w*\b"],
    "Linux": [r"\blinux\b", r"\bunix\b", r"\bbash\b"],
    "JavaScript": [r"\bjavascript\b", r"\bjs\b"],
    "TypeScript": [r"\btypescript\b"],
    "React": [r"\breact\b"],
    "Node.js": [r"\bnode\.?js\b", r"\bexpress\b"],
    "Java": [r"\bjava\b"],
    "C++": [r"\bc\+\+\b", r"\bcpp\b"],
    "Redis": [r"\bredis\b"],
    "GraphQL": [r"\bgraphql\b"],
    "Agile": [r"\bagile\b", r"\bscrum\b", r"\bkanban\b"],
}

# -------------------- UTILITIES --------------------

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9/\+\.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Try PyPDF2 first, fall back to pdfplumber for complex layouts."""
    # Try PyPDF2
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        if text.strip():
            return text
    except Exception:
        pass

    # Fallback: pdfplumber (handles columns, tables, complex layouts)
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
        return text
    except Exception:
        pass

    return ""


def extract_skills(text: str) -> set:
    found = set()
    for skill, patterns in SKILL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found.add(skill)
                break
    return found


def similarity_score(resume: str, jd: str) -> float:
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([resume, jd])
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)


def calculate_ats(similarity: float, missing: list, total_jd_skills: int) -> int:
    if total_jd_skills == 0:
        skill_weight = 100.0
    else:
        skill_weight = max(0, (total_jd_skills - len(missing)) / total_jd_skills) * 100
    return int(round(0.6 * skill_weight + 0.4 * similarity))


def score_message(score: int) -> str:
    if score >= 70:
        return "Strong match! Your resume aligns well with this role."
    elif score >= 40:
        return "Partial match. Consider adding the missing skills to improve your chances."
    else:
        return "Low match. Your resume needs updates to align with this job description."


# -------------------- PDF REPORT --------------------

def generate_pdf_report(score, ats, missing, resume_skills, jd_skills, report_path="resume_report.pdf"):
    doc = SimpleDocTemplate(
        report_path, pagesize=letter,
        rightMargin=inch, leftMargin=inch,
        topMargin=inch, bottomMargin=inch
    )
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=22, spaceAfter=6,
        textColor=colors.HexColor("#7c3aed")
    )
    story.append(Paragraph("Resume Analyzer Report", title_style))
    story.append(Spacer(1, 16))

    # Scores table
    score_data = [
        ["Metric", "Score", "Status"],
        ["Match Score", f"{score}%", "Good" if score >= 70 else ("Fair" if score >= 40 else "Poor")],
        ["ATS Compatibility", f"{ats}%", "Good" if ats >= 70 else ("Fair" if ats >= 40 else "Poor")],
    ]
    t = Table(score_data, colWidths=[2.5 * inch, 1.5 * inch, 1.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f3f0ff"), colors.white]),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Message
    story.append(Paragraph(score_message(score), styles["Normal"]))
    story.append(Spacer(1, 20))

    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"],
        fontSize=13, spaceAfter=6,
        textColor=colors.HexColor("#1e293b")
    )

    # Missing keywords
    story.append(Paragraph("Missing Keywords", heading_style))
    if missing:
        for kw in missing:
            story.append(Paragraph(f"• {kw}", styles["Normal"]))
    else:
        story.append(Paragraph("None — your resume covers all detected job requirements!", styles["Normal"]))
    story.append(Spacer(1, 16))

    # Resume skills
    story.append(Paragraph("Skills Found in Resume", heading_style))
    story.append(Paragraph(", ".join(resume_skills) if resume_skills else "None detected.", styles["Normal"]))
    story.append(Spacer(1, 16))

    # JD skills
    story.append(Paragraph("Skills Required by Job", heading_style))
    story.append(Paragraph(", ".join(jd_skills) if jd_skills else "None detected.", styles["Normal"]))

    doc.build(story)
    return report_path


# -------------------- ENDPOINTS --------------------

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    jd: str = Form(...)
):
    if not jd.strip():
        raise HTTPException(status_code=400, detail="Job description is empty")

    file_bytes = await file.read()
    resume_text = extract_text_from_pdf(file_bytes)

    if not resume_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from PDF. Make sure it's a text-based (not scanned) PDF."
        )

    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd)

    resume_skills = extract_skills(resume_clean)
    jd_skills = extract_skills(jd_clean)

    missing = sorted(jd_skills - resume_skills)
    similarity = similarity_score(resume_clean, jd_clean)
    ats = calculate_ats(similarity, missing, len(jd_skills))

    return {
        "score": int(round(similarity)),
        "ats": ats,
        "message": score_message(int(round(similarity))),
        "detected_resume_skills": sorted(resume_skills),
        "detected_jd_skills": sorted(jd_skills),
        "missing_skills": missing,
    }


@app.post("/download-report")
async def download_report(
    file: UploadFile = File(...),
    jd: str = Form(...)
):
    if not jd.strip():
        raise HTTPException(status_code=400, detail="Job description is empty")

    file_bytes = await file.read()
    resume_text = extract_text_from_pdf(file_bytes)

    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd)

    resume_skills = extract_skills(resume_clean)
    jd_skills = extract_skills(jd_clean)

    missing = sorted(jd_skills - resume_skills)
    similarity = similarity_score(resume_clean, jd_clean)
    ats = calculate_ats(similarity, missing, len(jd_skills))

    report_path = generate_pdf_report(
        score=int(round(similarity)),
        ats=ats,
        missing=missing,
        resume_skills=sorted(resume_skills),
        jd_skills=sorted(jd_skills),
    )

    return FileResponse(
        path=report_path,
        filename="Resume_Analyzer_Report.pdf",
        media_type="application/pdf"
    )