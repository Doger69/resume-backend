from fpdf import FPDF


def safe_text(text):
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_pdf(score, ats, missing, suggestions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, safe_text("Resume-Job Match Report"), ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, f"Match Score: {score}%", ln=True)
    pdf.cell(0, 10, f"ATS Score: {ats}%", ln=True)

    pdf.ln(5)
    pdf.cell(0, 10, "Missing Keywords:", ln=True)
    for k in missing:
        pdf.cell(0, 8, safe_text(f"- {k}"), ln=True)

    pdf.ln(5)
    pdf.cell(0, 10, "Suggestions:", ln=True)
    for s in suggestions:
        pdf.multi_cell(0, 8, safe_text(f"- {s}"))

    file_name = "resume_match_report.pdf"
    pdf.output(file_name)
    return file_name
