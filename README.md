# Data Dictionary

Source: [Student Performance Factors (Kaggle)](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

The dataset contains **19 explanatory variables** and **1 target variable** (`Exam_Score`).

## Numerical Variables

| Column | Description |
|---|---|
| `Hours_Studied` | Weekly study hours |
| `Attendance` | Class attendance rate (in %) |
| `Sleep_Hours` | Average hours of sleep per night |
| `Previous_Scores` | Scores obtained in previous exams |
| `Tutoring_Sessions` | Number of tutoring sessions attended per month |
| `Physical_Activity` | Weekly physical activity (in hours) |

## Nominal Categorical Variables (unordered)

Encoding used: **one-hot encoding**.

| Column | Description | Values |
|---|---|---|
| `Gender` | Gender | Male / Female |
| `School_Type` | Type of school | Public / Private |
| `Extracurricular_Activities` | Participation in extracurricular activities | Yes / No |
| `Internet_Access` | Internet access | Yes / No |
| `Learning_Disabilities` | Presence of learning disabilities | Yes / No |
| `Peer_Influence` | Peer influence | Positive / Neutral / Negative |

## Ordinal Categorical Variables (ordered)

Encoding used: **ordinal encoding** (0, 1, 2 according to ascending order).

| Column | Description | Value Order |
|---|---|---|
| `Parental_Involvement` | Parental involvement | Low < Medium < High |
| `Access_to_Resources` | Access to educational resources | Low < Medium < High |
| `Motivation_Level` | Motivation level | Low < Medium < High |
| `Family_Income` | Family income | Low < Medium < High |
| `Teacher_Quality` | Perceived teacher quality (missing values) | Low < Medium < High |
| `Parental_Education_Level` | Parents' education level (missing values) | High School < College < Postgraduate |
| `Distance_from_Home` | Distance from home to school (missing values) | Near < Moderate < Far |

## Target Variable

| Column | Description |
|---|---|
| `Exam_Score` | Final exam score (continuous value) |

## Missing Values

The columns `Teacher_Quality`, `Parental_Education_Level`, and `Distance_from_Home` contain missing values. They are replaced with the **mode** (most frequent value) of each column.