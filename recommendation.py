
from __future__ import annotations

import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Candidate locations for the CSV file: same folder as this module, and a
# "data" subfolder (matches this project's actual layout: data/course.csv).
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV_CANDIDATES = [
    os.path.join(_MODULE_DIR, "data", "course.csv"),
    os.path.join(_MODULE_DIR, "course.csv"),
]


class CourseRecommender:
    """Recommends similar courses using TF-IDF + Cosine Similarity."""

    def __init__(self, data: pd.DataFrame | None = None):
        self.df = data if data is not None else self._default_dataset()
        self._build_model()

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    @staticmethod
    def _default_dataset() -> pd.DataFrame:
        """Load courses from course.csv, checked in the data/ subfolder
        first, then next to this file."""
        csv_path = next((p for p in _CSV_CANDIDATES if os.path.exists(p)), None)

        if csv_path is None:
            searched = "\n".join(f"  - {p}" for p in _CSV_CANDIDATES)
            raise FileNotFoundError(
                f"course.csv not found. Searched:\n{searched}"
            )

        df = pd.read_csv(csv_path)

        # Normalize column names: strip stray whitespace so headers like
        # "Course_Name " or " Description" still match.
        df.columns = [str(c).strip() for c in df.columns]

        required = ["Course_Name", "Category", "Level", "Description"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"course.csv is missing required column(s): {missing}. "
                f"Columns found in the file: {list(df.columns)}. "
                "Check for typos, extra spaces, or different capitalization "
                "in the CSV header row."
            )

        # Basic cleanup: drop rows with missing essentials, reset index
        df = df.dropna(subset=["Course_Name", "Description"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    def _build_model(self) -> None:
        corpus = (
            self.df["Category"] + " " +
            self.df["Level"] + " " +
            self.df["Description"]
        )
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def get_course_list(self) -> list[str]:
        return self.df["Course_Name"].tolist()

    def recommend(self, course_name: str, top_n: int = 5) -> list[dict]:
        if course_name not in self.df["Course_Name"].values:
            return []

        idx = self.df.index[self.df["Course_Name"] == course_name][0]
        scores = list(enumerate(self.similarity_matrix[idx]))

        # Exclude the course itself, sort by similarity descending
        scores = [s for s in scores if s[0] != idx]
        scores.sort(key=lambda x: x[1], reverse=True)

        top_scores = scores[:top_n]

        results = []
        for i, score in top_scores:
            row = self.df.iloc[i]
            results.append({
                "Course_Name": row["Course_Name"],
                "Category": row["Category"],
                "Level": row["Level"],
                "Description": row["Description"],
                "Similarity": round(float(score) * 100, 1),
            })
        return results