# 🎓 AI Course Recommendation System

An AI-powered Course Recommendation System built using **Python**, **Streamlit**, **TF-IDF Vectorization**, and **Cosine Similarity**. The system recommends courses that are most similar to a selected course by analyzing textual information such as the course name, category, level, and description.

---

## 📌 Project Overview

Choosing the right learning path can be challenging due to the large number of available courses. This project uses Natural Language Processing (NLP) techniques to recommend relevant courses based on textual similarity.

The recommendation engine converts course descriptions into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)** and measures similarity using **Cosine Similarity**.

---

## 🚀 Features

* AI-powered course recommendation
* TF-IDF Vectorization
* Cosine Similarity Algorithm
* Interactive Streamlit Web Interface
* Clean and responsive design
* Beginner to Advanced course recommendations
* Fast recommendation generation
* Easy to customize with new datasets

---

## 🛠 Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorizer
* Cosine Similarity

---

## 📂 Project Structure

```text
AI_Recommendation_System/
│
├── app.py
├── recommendation.py
├── utils.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
│
├── data/
│   └── courses.csv
│
└── assets/
    └── screenshots/
```

---

## 📊 Dataset

The dataset contains:

* Course Name
* Category
* Difficulty Level
* Course Description

The recommendation engine combines these fields to calculate similarity between courses.

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/AI_Recommendation_System.git
```

### Move into the project directory

```bash
cd AI_Recommendation_System
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. Load the course dataset.
2. Combine course information into a single text field.
3. Convert the text into TF-IDF vectors.
4. Compute cosine similarity between all courses.
5. Recommend the top similar courses based on the selected course.

---

## 📸 Screenshots
![alt text](<Screenshot 2026-07-23 190314-1.png>)
![alt text](<Screenshot 2026-07-23 190338.png>)

---

## 🔮 Future Improvements

* User authentication
* Personalized recommendations based on user history
* Deep Learning recommendation models
* Content-based + Collaborative Filtering hybrid system
* Database integration (MySQL/PostgreSQL)
* Course ratings and reviews
* Course search with filters
* Deployment using Docker and Cloud

---

## 👨‍💻 Author

ABHISHEK THUVVARA

B.Tech Artificial Intelligence Student

LinkedIn:https://www.linkedin.com/in/thuvvara-abhishek-01768b362/

---
📜 License
This project is created for educational and internship purposes under the DecodeLabs AI Internship Program.
