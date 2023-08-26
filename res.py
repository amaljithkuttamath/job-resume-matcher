import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import time
import requests
import pandas as pd
import datetime

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")

SAMPLE = 7

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the text
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)

def fetch_data(query, page, num_pages):
    url = "https://jsearch.p.rapidapi.com/search"

    querystring = {"query": query, "page": page, "num_pages": num_pages}

    headers = {
        "X-RapidAPI-Key": "1b4f9da3f3mshc19729cb573c6f4p12318bjsnfe4c509c9fa5",
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status code {response.status_code}")

def save_data_to_csv(data, query):
    df = pd.DataFrame(data['data'])

    # Add today's date and query to the filename
    today = datetime.date.today().strftime("%Y-%m-%d")
    filename = f"data_{query.replace(' ', '_')}_{today}.csv"

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def save_jobs_to_csv():
    query = ["Data Analyst in USA"]
    page = "10"
    num_pages = "10"
    for q in query: 
        data = fetch_data(str(q), page, num_pages)
        save_data_to_csv(data, str(q))

def get_resume_data():
    return pd.read_csv('UpdatedResumeDataSet.csv').sample(n=SAMPLE)

# def get_job_data():
#     # create an empty list to store all dataframes
#     dfs = []

#     # loop through all files in the directory
#     for filename in os.listdir('.\\Project'):
#         if filename.startswith('data_') and filename.endswith('.csv'):
#             # read the csv file and append to the list of dataframes
#             temp_df = pd.read_csv("Project\\"+filename)
#             dfs.append(temp_df)

#     return pd.concat(dfs, ignore_index=True)

def get_job_data():
    job_df = pd.read_csv("data_Data_Analyst_in_USA_2023-05-13.csv") 
    # fetch data if job_required_experience,job_required_skills,job_required_education exists
    job_df = job_df[job_df['job_required_experience'].notna() & job_df['job_required_skills'].notna() & job_df['job_required_education'].notna()]
    # job_df.dropna(inplace=True)
    # drop duplicates
    # job_df = job_df.drop_duplicates(subset=['job_title', 'job_company', 'job_location', 'job_description'], keep='first')
    # # drop rows with empty job_description
    job_df = job_df[job_df['job_description'] != '']
    # # drop rows with empty job_required_experience
    job_df = job_df[job_df['job_required_experience'] != '']
    # # drop rows with empty job_required_skills
    job_df = job_df[job_df['job_required_skills'] != '']
    
    return job_df

def extract_top_keywords(text, n=10, domain_specific_vocab=None):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.pos_ not in ['PRON', 'DET', 'ADP', 'AUX']]

    if domain_specific_vocab:
        tokens.extend(domain_specific_vocab)

    filtered_text = " ".join(tokens)

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
    count_matrix = vectorizer.fit_transform([filtered_text])
    word_freq = dict(zip(vectorizer.get_feature_names_out(), count_matrix.toarray().sum(axis=0)))
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:n]

    return [word for word, freq in sorted_word_freq]

def generate_suggestions(resume_text, job_descriptions, n=10, domain_specific_vocab=None):
    resume_keywords = extract_top_keywords(resume_text, n, domain_specific_vocab)

    combined_df = pd.DataFrame(columns=['Keyword', 'Frequency'])

    for job in job_descriptions:
        job_keywords = extract_top_keywords(job, n, domain_specific_vocab)
        job_word_freq = [job.count(word) for word in job_keywords]
        combined_df = pd.concat([combined_df, pd.DataFrame({'Keyword': job_keywords, 'Frequency': job_word_freq})], ignore_index=True)

    combined_df = combined_df.groupby('Keyword').sum().sort_values(by='Frequency', ascending=False).reset_index()

    suggestions = [keyword for keyword in combined_df['Keyword'] if keyword not in resume_keywords][:n]

    # Extract relevant phrases and named entities
    job_entities = []
    job_titles = []
    for job_description in job_descriptions:
        doc = nlp(job_description)
        entities = [entity.text.lower() for entity in doc.ents if entity.label_ in ["ORG", "PRODUCT", "GPE", "LANGUAGE", "SKILL"]]
        phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
        titles = [token.text.lower() for token in doc if token.pos_ == "NOUN"]

        job_entities.extend(entities)
        job_entities.extend(phrases)
        job_titles.extend(titles)

    # Count the occurrences of each entity and filter suggestions based on their frequency
    entity_count = Counter(job_entities)
    title_count = Counter(job_titles)

    # ?Add the top n entities and titles to suggestions
    top_entities = [entity for entity, freq in entity_count.most_common(n) if entity not in resume_keywords]
    top_titles = [title for title, freq in title_count.most_common(n) if title not in resume_keywords]

    suggestions.extend(top_entities)
    suggestions.extend(top_titles)

    return list(dict.fromkeys(suggestions))[:n]

def extract_entities(text):
    doc = nlp(text)
    return [
        (ent.text.lower(), ent.label_)
        for ent in doc.ents
        if ent.label_ in ["ORG", "PRODUCT", "GPE", "LANGUAGE", "SKILL"]
    ]

def extract_experience(resume_text):
    # search for the pattern "Experience - x months/years" and return the number x
    exp = re.search(r'(\d+)\s*(month|year)', resume_text, re.I)
    if exp:
        return int(exp.group(1))
    return 0

def extract_skills(resume_text):
    skills = ["python", "java", "machine learning", "sql", "javascript", "jquery", "css", "html", "angular", "docker", "git", "flask", "kafka", "logstash"]
    return [
        skill
        for skill in skills
        if re.search(r'\b' + skill + r'\b', resume_text, re.I)
    ]

def extract_education(resume_text):
    # search for the pattern "B.E", "PhD", "Master's" etc.
    if re.search(r'\bB\.E\b', resume_text, re.I):
        return 'B.E'
    elif re.search(r'\bPhD\b', resume_text, re.I):
        return 'PhD'
    elif re.search(r'\bMaster\b', resume_text, re.I):
        return 'Master'
    else:
        return 'Not Specified'

def match_resume_with_jobs(resume_text, job_data):
    # Extract named entities (skills, education, experience) from the resume
    candidate_skills = set(extract_skills(resume_text))
    candidate_experience = extract_experience(resume_text)
    candidate_education = extract_education(resume_text)

    # Initialize the vectorizer and compute the tf-idf matrix
    vectorizer = TfidfVectorizer(stop_words="english")
    all_documents = [preprocess_text(resume_text)] + job_data["job_description"].apply(preprocess_text).tolist()
    tfidf_matrix = vectorizer.fit_transform(all_documents)

    # Compute the cosine similarity between the resume and job descriptions
    resume_vector = tfidf_matrix[0:1]
    job_vectors = tfidf_matrix[1:]
    cosine_similarities = cosine_similarity(resume_vector, job_vectors).flatten()

    # Add similarity scores to the job_data DataFrame
    job_data["similarity"] = cosine_similarities

    # Sort the jobs by similarity score
    sorted_jobs = job_data.sort_values(by="similarity", ascending=False).reset_index(drop=True)

    scores = []

    # Iterate over each job
    for index, job in sorted_jobs.iterrows():
        job_required_skills = set(job['job_required_skills'].lower().split(","))
        job_required_experience = job['job_required_experience']
        job_required_education = job['job_required_education'].lower()

        try:
            job_required_experience_dict = eval(job_required_experience)
            job_required_experience = job_required_experience_dict.get('required_experience_in_months', 0)
            job_required_experience = (
                int(job_required_experience) if job_required_experience else 0
            )
        except (ValueError, SyntaxError):
            print(f"Invalid experience value for job: {job_required_experience}. Skipping this job.")
            continue

        # Calculate skill match score
        skills_match_score = len(candidate_skills & job_required_skills) / len(job_required_skills)

        # Calculate experience match score
        if candidate_experience >= job_required_experience:
            exp_match_score = 1
        else:
            exp_match_score = candidate_experience / job_required_experience

        # Calculate education match score
        edu_match_score = 1 if candidate_education.lower() == job_required_education else 0.5  # assuming some score if education doesn't match exactly

        # Combine scores (you can assign weights as per your requirement)
        total_score = (skills_match_score + exp_match_score + edu_match_score) / 3

        scores.append({'Job': job['job_title'], 'Score': total_score})

    sorted_jobs["Score"] = pd.Series([score['Score'] for score in scores])

    return sorted_jobs.sort_values(by="Score", ascending=False).reset_index(drop=True)

def main():
    # Load the resume dataset 
    resume_data = get_resume_data()
    # reset index and rename it to resume_id
    resume_data = resume_data.reset_index().rename(columns={'index': 'Resume ID'})
    
    result_df = pd.DataFrame(columns=['Job ID', 'Suggestions', 'Resume ID', 'Match Score', 'Apply Link', 'Job Skills'])

    # Iterate through all the resumes in the dataset
    for index, row in resume_data.iterrows():
        print(f"Processing resume {index + 1}/{len(resume_data)}")
        resume_text = row["Resume"]
        resume_id = row["Resume ID"]
        job_data = get_job_data().sample(SAMPLE)
        matched_jobs = match_resume_with_jobs(preprocess_text(resume_text), job_data)
        top_matched_jobs = matched_jobs.head(10)
        print(top_matched_jobs)
        for _, job in top_matched_jobs.iterrows():
            job_id = job['job_id']
            job_description = job['job_description']
            match_score = job['similarity']
            apply_link = job['job_apply_link']
            score = job['Score']
            job_skills = job['job_required_skills']
            job_experience = job['job_required_experience']
            job_education = job['job_required_education']

            suggestions = generate_suggestions(resume_text, [job_description])

            resume_skills = extract_skills(resume_text)
            resume_experience = extract_experience(resume_text)

            temp_df = pd.DataFrame({
                'Job ID': [job_id],
                'Suggestions': [suggestions],
                'Resume ID': [resume_id],
                'Match Score': [match_score],
                'Apply Link': [apply_link],
                'Resume Experience': [resume_experience],
                'Resume Skills': [resume_skills],
                'Score': [score],
                'Job Experience': [job_experience],
                'Job Education': [job_education],
                'Job Skills': [job_skills]
                })
            result_df = pd.concat([result_df, temp_df], ignore_index=True)

    result_df.to_csv(f'result_{datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}.csv', index=False)

def visualize_top_suggestions(result_df, n=10):
    keyword_counts = {}

    for _ , row in result_df.iterrows():
        suggestions = row['Suggestions']
        for keyword in suggestions:
            if keyword in keyword_counts:
                keyword_counts[keyword] += 1
            else:
                keyword_counts[keyword] = 1

    sorted_keyword_counts = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)

    keywords, counts = zip(*sorted_keyword_counts[:n])
    plt.barh(keywords, counts)

    plt.xlabel("Frequency")
    plt.ylabel("Keywords")
    plt.title("Top 10 Suggested Keywords Across Matched Jobs")
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    main()
