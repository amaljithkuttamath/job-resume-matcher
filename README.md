# Resume Matcher and Job Suggestion Tool Overview

This Python script offers a comprehensive solution for matching resumes with job descriptions and providing suggestions to enhance resumes for better alignment with potential job opportunities. It utilizes Natural Language Processing (NLP) techniques, TF-IDF, and cosine similarity for effective matching and suggestion generation based on common keywords in job descriptions.

## How it Works

- **Data Fetching:** The script fetches job data via an API request, including job descriptions, required skills, and experience.
- **Resume Processing:** Processes a set of resumes and matches each with the fetched jobs based on similarity.
- **Suggestions:** Provides tailored suggestions for each resume based on frequent keywords in matched job descriptions.
- **Visualization:** Displays a bar chart of top suggested keywords across all matched jobs.

## Usage

### Installation

Install the necessary libraries with pip:

`pip install pandas sklearn nltk matplotlib spacy requests`


Ensure you have all necessary data files in the script's directory, including resumes and job descriptions.

### Execution

Run the script with:

`python res.py`


The script outputs a CSV file with match results and suggestions, and displays a bar chart of top suggested keywords.

## Functions Overview

- `preprocess_text(text)`: Converts text to lowercase, tokenizes, removes stop words, and lemmatizes.
- `fetch_data(query, page, num_pages)`: Fetches job data using the RapidAPI job search API.
- `save_data_to_csv(data, query)`: Saves fetched data to a CSV file.
- `get_resume_data()`, `get_job_data()`: Load resume and job data from CSV files.
- `extract_top_keywords(text, n=10, domain_specific_vocab=None)`: Extracts top *n* keywords from text.
- `generate_suggestions(resume_text, job_descriptions, n=10, domain_specific_vocab=None)`: Generates resume enhancement suggestions.
- `extract_entities(text)`: Extracts named entities from text.
- `extract_experience(resume_text)`, `extract_skills(resume_text)`, `extract_education(resume_text)`: Extract relevant resume information.
- `match_resume_with_jobs(resume_text, job_data)`: Matches resumes with jobs based on similarity.
- `main()`: Coordinates the entire process.
- `visualize_top_suggestions(result_df, n=10)`: Visualizes top *n* suggested keywords.

## Limitations

Note: This script uses a sample of 7 resumes and jobs for brevity. Adjustments may be required for larger datasets.

## Dependencies

The script relies on several Python libraries: pandas, sklearn, nltk, matplotlib, spacy, and requests. Install these before running the script.
