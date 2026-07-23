import os
import re
import pandas as pd


def load_dataset(data_path="data/courses.csv"):
    """
    Load the course dataset safely and validate essential columns.
    
    Parameters
    ----------
    data_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at path: '{data_path}'")

    df = pd.read_csv(data_path)

    required_columns = ["Course_Name", "Category", "Level", "Description"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    return df.fillna("")


def clean_text(text):
    """
    Clean text for vectorization while preserving programming language 
    symbols like '+' (C++) and '#' (C#).
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    # Allow alphanumeric characters, spaces, and tech symbols like +, #
    text = re.sub(r"[^a-zA-Z0-9\+#\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(df):
    """
    Prepares the DataFrame for recommendation models.
    Preserves original columns for user display and creates a 
    'Combined_Text' feature vector for cleaning and TF-IDF.
    """
    df = df.copy()

    # Create a unified string representation for feature processing
    raw_combined = (
        df["Course_Name"].astype(str) + " " +
        df["Category"].astype(str) + " " +
        df["Level"].astype(str) + " " +
        df["Description"].astype(str)
    )

    df["Cleaned_Text"] = raw_combined.apply(clean_text)
    return df


def validate_course(df, course_name):
    """
    Check whether a course title exists in the dataset (case-insensitive).
    """
    if not course_name or not str(course_name).strip():
        return False
    
    clean_target = course_name.strip().lower()
    return clean_target in df["Course_Name"].str.lower().values


def get_categories(df):
    """Return sorted unique course categories."""
    return sorted(df["Category"].dropna().unique())


def get_levels(df):
    """Return sorted unique difficulty levels."""
    return sorted(df["Level"].dropna().unique())


def filter_courses(df, category=None, level=None):
    """
    Filter dataset by category and/or level while maintaining clean indices.
    """
    filtered = df.copy()

    if category and category.strip():
        filtered = filtered[
            filtered["Category"].str.lower() == category.strip().lower()
        ]

    if level and level.strip():
        filtered = filtered[
            filtered["Level"].str.lower() == level.strip().lower()
        ]

    return filtered.reset_index(drop=True)


def dataset_summary(df):
    """
    Return basic dataset statistics.
    """
    return {
        "Total Courses": len(df),
        "Categories": df["Category"].nunique(),
        "Levels": df["Level"].nunique()
    }


if __name__ == "__main__":
    # Smoke Test Demonstration
    sample_data = {
        "Course_Name": ["Python for Beginners", "C++ Programming", "React.js"],
        "Category": ["Programming", "Programming", "Web Development"],
        "Level": ["Beginner", "Intermediate", "Intermediate"],
        "Description": ["Learn Python 3", "Master C++ OOP concepts", "Build React UI components"]
    }
    
    test_df = pd.DataFrame(sample_data)
    processed_df = preprocess_dataframe(test_df)
    
    print("Dataset Summary:", dataset_summary(processed_df))
    print("Course Exists 'C++ Programming':", validate_course(processed_df, "c++ programming"))
    print("\nProcessed Features:\n", processed_df[["Course_Name", "Cleaned_Text"]])