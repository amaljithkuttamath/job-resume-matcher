# job-resume-matcher

Resume Matcher and Job Suggestion Tool
Overview
This python script provides a comprehensive solution for matching resumes with job descriptions, as well as providing suggestions for enhancing resumes to better match with potential job opportunities. It leverages Natural Language Processing (NLP) techniques, TF-IDF and cosine similarity for matching resumes and job descriptions, and provides suggestions based on the most common keywords found in the job descriptions.

How it works
The script fetches job data via an API request. The job data includes fields such as job descriptions, required skills, and required experience.

It then processes a set of resumes and matches each resume with the fetched jobs based on the similarity between the resume and the job descriptions.

After matching, the script provides suggestions for each resume based on the most frequent keywords found in the matched job descriptions.

The script also provides a visualization of the top suggested keywords across all matched jobs.

Usage
Install the necessary libraries. These include pandas, sklearn, nltk, matplotlib, spacy, requests and others. You can install these using pip:


pip install pandas sklearn nltk matplotlib spacy requests


Make sure you have the necessary data files in the same directory as the script. This includes the resumes and job descriptions.

Run the script:

python res.py


The script will output a CSV file containing the matching results and suggestions. It will also display a bar chart showing the top suggested keywords.
Functions
The script contains several helper functions. Here's a brief overview:

preprocess_text(text): Preprocesses the input text by converting it to lowercase, tokenizing, removing stop words and lemmatizing.

fetch_data(query, page, num_pages): Fetches job data using the RapidAPI job search API.

save_data_to_csv(data, query): Saves the fetched data to a CSV file.

get_resume_data(): Loads resume data from a CSV file.

get_job_data(): Loads job data from a CSV file.

extract_top_keywords(text, n=10, domain_specific_vocab=None): Extracts the top n keywords from the input text.

generate_suggestions(resume_text, job_descriptions, n=10, domain_specific_vocab=None): Generates suggestions based on the most frequent keywords found in the job descriptions.

extract_entities(text): Extracts named entities such as organizations, products, locations, languages, and skills from the input text.

extract_experience(resume_text), extract_skills(resume_text), extract_education(resume_text): Extract relevant information from the resume text.

match_resume_with_jobs(resume_text, job_data): Matches a resume with jobs based on the similarity between the resume and the job descriptions.

main(): The main function that coordinates the entire process.

visualize_top_suggestions(result_df, n=10): Provides a visualization of the top n suggested keywords.

Limitations
Please note that this script uses a sample of 7 resumes and jobs for the sake of brevity. You may need to adjust the sample size or modify the script to handle larger datasets as per your requirements.

Dependencies
This script depends on several Python libraries including pandas, sklearn, nltk, matplotlib, spacy, and requests. Make sure to install these libraries before running the script.