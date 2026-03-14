import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(__file__)
SCHOLARSHIP_PATH = os.path.join(BASE_DIR, "../../data/scholarship_training_data.csv")

with open(SCHOLARSHIP_PATH) as f:
    universities = pd.read_csv(f)

model = joblib.load("src/models/scholarship_model.pkl")

# clean column names
universities.columns = universities.columns.str.strip()

# drop invalid rows
universities = universities.dropna(subset=["University", "Location", "Majors"])
universities = universities[universities["Majors"].str.strip() != ""]

universities["quota_norm"] = universities["Quota"] / universities["Quota"].max()


grade_map = {
    "A":5,
    "B":4,
    "C":3,
    "D":2,
    "E":1
}

def recommend_universities(student_grade, major, province):

    print(f"Recommending universities...{student_grade}, {major}, {province}")

    results = []

    grade_score = grade_map.get(student_grade.upper(), 0)

    province = province.lower()
    major = major.lower()

    for _, uni in universities.iterrows():

        if uni["Location"].lower() != province:
            continue

        major_list = [m.strip().lower() for m in uni["Majors"].split("|")]

        if major not in major_list:
            continue

        scholarship = uni[f"Grade{student_grade.upper()}"]

        if scholarship == 0:
            continue

        features = pd.DataFrame([{
            "student_grade": grade_score,
            "grade_diff": grade_score - 3,
            "scholarship_norm": scholarship / 100,
            "quota_norm": uni["quota_norm"],
            "entrance_exam": 1 if uni["EntranceExam"] == "Yes" else 0,
            "major_match": 1,
            "province_match": 1
        }])

        score = model.predict_proba(features)[0][1]

        results.append({
            "university": uni["University"],
            "location": uni["Location"],
            "scholarship": scholarship,
            "quota": uni["Quota"],
            "entrance_exam": uni["EntranceExam"],
            "score": round(score * 100, 2)
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:5]

