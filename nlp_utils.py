import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# REQUIRED DOWNLOADS
nltk.download("punkt")
nltk.download("stopwords")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    return " ".join(w for w in words if w not in stop_words)


def calculate_similarity(resume_text, job_description):
    resume = remove_stopwords(clean_text(resume_text))
    job = remove_stopwords(clean_text(job_description))

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume, job])

    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
    return round(score, 2)


def keyword_gap_analysis(resume_text, job_text, top_n=20):
    resume_words = set(remove_stopwords(clean_text(resume_text)).split())
    job_words = remove_stopwords(clean_text(job_text)).split()

    freq = nltk.FreqDist(job_words)
    important = [w for w, _ in freq.most_common(top_n)]

    return [kw for kw in important if kw not in resume_words]


def ats_score(similarity_score, missing_keywords, total_keywords=20):
    keyword_score = ((total_keywords - len(missing_keywords)) / total_keywords) * 100
    return round((0.6 * keyword_score) + (0.4 * similarity_score), 2)


def resume_suggestions(score, missing):
    tips = []

    if score < 40:
        tips.append("Resume has low alignment with job requirements.")
    elif score < 70:
        tips.append("Resume partially matches the job role.")
    else:
        tips.append("Resume is well-aligned with the job.")

    if missing:
        tips.append("Add these missing skills: " + ", ".join(missing[:8]))

    tips.append("Use action verbs with measurable achievements.")
    tips.append("Keep formatting ATS-friendly (no tables, graphics, or columns).")

    return tips
