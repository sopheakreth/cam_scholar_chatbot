import pandas as pd
import numpy as np
import random
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(__file__)
SCHOLARSHIP_PATH = os.path.join(BASE_DIR, "../../data/scholarship_training_data.csv")


grade_map = {
    "A":5,
    "B":4,
    "C":3,
    "D":2,
    "E":1
}

def train_model():

    with open(SCHOLARSHIP_PATH) as f:
        df = pd.read_csv(f)

    # Clean Data
    col = ['Quota', 'Majors']
    df[col] = df[col].replace(0, np.nan)

    df.isnull().sum()
    df = df.dropna()

    # Convert numeric columns
    numeric_cols = ["GradeA", "GradeB", "GradeC", "GradeE", "Quota"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace missing numeric values
    df[numeric_cols] = df[numeric_cols].fillna(0)
    # Remove universities where all scholarships are 0
    df = df[
        (df[["GradeA", "GradeB", "GradeC", "GradeE"]].sum(axis=1) > 0)
    ]
    # Final Clean Dataset Check
    df.reset_index(drop=True, inplace=True)

    # Convert Entrance Exam
    exam_map = {"No": 0, "Yes": 1}
    df["EntranceExam"] = df["EntranceExam"].map(exam_map).fillna(0)

    majors_all = set()

    for majors in df["Majors"]:
        majors_all.update([m.strip() for m in majors.split("|")])

    majors_all = list(majors_all)

    provinces = df["Location"].unique()

    # Normalize Quota
    df["quota_norm"] = df["Quota"] / df["Quota"].max()

    # Generate Machine Learning
    training_rows = []

    for _, uni in df.iterrows():

        for grade in ["A","B","C","D","E"]:

            scholarship = uni[f"Grade{grade}"]

            if scholarship == 0:
                continue

            for _ in range(20):

                student_major = random.choice(majors_all)
                student_province = random.choice(provinces)

                major_list = uni["Majors"].split("|")

                major_match = 1 if student_major in major_list else 0
                province_match = 1 if student_province == uni["Location"] else 0

                student_grade_score = grade_map[grade]

                grade_diff = student_grade_score - 3

                scholarship_norm = scholarship / 100
                quota_norm = uni["quota_norm"]

                entrance_exam = 1 if uni["EntranceExam"] == "Yes" else 0

                match = 0
                if scholarship >= 50 and major_match == 1:
                    match = 1

                training_rows.append({
                    "student_grade": student_grade_score,
                    "grade_diff": grade_diff,
                    "scholarship_norm": scholarship_norm,
                    "quota_norm": quota_norm,
                    "entrance_exam": entrance_exam,
                    "major_match": major_match,
                    "province_match": province_match,
                    "match": match
                })

    df = pd.DataFrame(training_rows)

    X = df.drop("match", axis=1)
    y = df["match"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)

    print("Predicting is completed...")

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Among {len(y_test):d}, our ML model classified {accuracy * 100:.2f}% correctly")

    joblib.dump(model, "src/models/scholarship_model.pkl")

    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()