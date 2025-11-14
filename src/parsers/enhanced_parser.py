"""
Enhanced Resume Parser

Improved parsing with better skill extraction, experience calculation,
and section identification using SpaCy and regex patterns.
"""

import re
import spacy
from typing import Dict, List, Optional, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EnhancedResumeParser:
    """
    Enhanced resume parser with improved extraction capabilities.
    """

    def __init__(self, spacy_model='en_core_web_sm'):
        """
        Initialize parser.

        Args:
            spacy_model: SpaCy model to use
        """
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"✓ Loaded SpaCy model: {spacy_model}")
        except OSError:
            logger.error(f"SpaCy model '{spacy_model}' not found. Install with: python -m spacy download {spacy_model}")
            raise

        # Comprehensive skill database (expand as needed)
        self.skills_database = self._load_skills_database()

        # Education keywords
        self.education_keywords = {
            'phd', 'ph.d', 'doctorate', 'doctor', 'postdoc',
            'master', "master's", 'ms', 'm.s', 'mba', 'm.b.a',
            'bachelor', "bachelor's", 'bs', 'b.s', 'ba', 'b.a', 'b.e', 'b.tech',
            'associate', "associate's", 'diploma', 'certificate', 'certification'
        }

        # Experience patterns
        self.experience_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)',
            r'(?:experience|exp)(?:\s+of)?\s+(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\s*-\s*(\d+)\s*(?:years?|yrs?)',
        ]

    def _load_skills_database(self) -> Dict[str, List[str]]:
        """Load comprehensive skills database by category."""
        return {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'swift',
                'kotlin', 'scala', 'r', 'matlab', 'php', 'perl', 'shell', 'bash', 'powershell', 'sql',
                'html', 'css', 'sass', 'less'
            ],
            'ml_ai': [
                'machine learning', 'deep learning', 'neural networks', 'ai', 'artificial intelligence',
                'nlp', 'natural language processing', 'computer vision', 'reinforcement learning',
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'xgboost', 'lightgbm',
                'hugging face', 'transformers', 'bert', 'gpt', 'llm', 'large language models'
            ],
            'data_science': [
                'data analysis', 'data science', 'statistics', 'statistical analysis',
                'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
                'jupyter', 'data visualization', 'exploratory data analysis', 'eda',
                'feature engineering', 'data mining', 'predictive modeling'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
                'dynamodb', 'neo4j', 'oracle', 'sql server', 'sqlite', 'mariadb',
                'nosql', 'database design', 'database optimization', 'sql'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'google cloud', 'cloud computing',
                'docker', 'kubernetes', 'k8s', 'ci/cd', 'jenkins', 'github actions',
                'terraform', 'ansible', 'devops', 'microservices', 'serverless',
                'lambda', 'ec2', 's3', 'cloudformation'
            ],
            'web_frameworks': [
                'react', 'angular', 'vue', 'vue.js', 'next.js', 'node.js', 'express',
                'django', 'flask', 'fastapi', 'spring', 'spring boot', 'rails', 'ruby on rails',
                'asp.net', '.net', 'rest api', 'graphql', 'websockets'
            ],
            'data_engineering': [
                'spark', 'apache spark', 'hadoop', 'hive', 'kafka', 'airflow', 'etl',
                'data pipeline', 'data warehousing', 'bigquery', 'snowflake', 'redshift',
                'databricks', 'data lake', 'stream processing', 'batch processing'
            ],
            'tools': [
                'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence',
                'slack', 'visual studio code', 'intellij', 'pycharm', 'eclipse',
                'postman', 'swagger', 'linux', 'unix', 'windows'
            ],
            'soft_skills': [
                'leadership', 'team work', 'communication', 'problem solving',
                'critical thinking', 'agile', 'scrum', 'project management',
                'stakeholder management', 'mentoring', 'collaboration'
            ]
        }

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills by category from resume text.

        Args:
            text: Resume text

        Returns:
            Dict mapping category to list of found skills
        """
        text_lower = text.lower()
        found_skills = defaultdict(list)

        for category, skills_list in self.skills_database.items():
            for skill in skills_list:
                # Use word boundaries for more accurate matching
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills[category].append(skill)

        # Convert to regular dict
        return dict(found_skills)

    def extract_all_skills_flat(self, text: str) -> List[str]:
        """
        Extract all skills as a flat list.

        Args:
            text: Resume text

        Returns:
            List of all found skills
        """
        skills_by_category = self.extract_skills(text)
        all_skills = []

        for category_skills in skills_by_category.values():
            all_skills.extend(category_skills)

        return sorted(set(all_skills))

    def extract_experience_years(self, text: str) -> Optional[int]:
        """
        Extract years of experience from resume.

        Args:
            text: Resume text

        Returns:
            Number of years of experience, or None
        """
        text_lower = text.lower()

        for pattern in self.experience_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                # If range (e.g., "5-7 years"), take the average
                if isinstance(matches[0], tuple):
                    years = [int(y) for y in matches[0] if y.isdigit()]
                    if years:
                        return int(sum(years) / len(years))
                else:
                    # Single number
                    try:
                        return int(matches[0])
                    except (ValueError, IndexError):
                        continue

        return None

    def extract_education(self, text: str) -> List[str]:
        """
        Extract education degrees from resume.

        Args:
            text: Resume text

        Returns:
            List of found degrees
        """
        text_lower = text.lower()
        found_degrees = []

        # More specific degree patterns
        degree_patterns = [
            (r'\b(ph\.?d|doctorate|doctor of philosophy)\b', 'PhD'),
            (r'\b(m\.?s\.?|master of science)\b', 'Master of Science'),
            (r'\b(m\.?b\.?a\.?|master of business administration)\b', 'MBA'),
            (r'\b(master|master\'s)\b', 'Master\'s Degree'),
            (r'\b(b\.?s\.?|bachelor of science)\b', 'Bachelor of Science'),
            (r'\b(b\.?a\.?|bachelor of arts)\b', 'Bachelor of Arts'),
            (r'\b(b\.?e\.?|bachelor of engineering)\b', 'Bachelor of Engineering'),
            (r'\b(b\.?tech|bachelor of technology)\b', 'Bachelor of Technology'),
            (r'\b(associate|associate\'s)\b', 'Associate Degree'),
        ]

        for pattern, degree_name in degree_patterns:
            if re.search(pattern, text_lower):
                if degree_name not in found_degrees:
                    found_degrees.append(degree_name)

        return found_degrees

    def extract_contact_info(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract contact information.

        Args:
            text: Resume text

        Returns:
            Dict with email, phone, linkedin
        """
        contact_info = {
            'email': None,
            'phone': None,
            'linkedin': None
        }

        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info['email'] = email_match.group()

        # Phone (US format)
        phone_pattern = r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact_info['phone'] = ''.join(phone_match.groups())

        # LinkedIn
        linkedin_pattern = r'linkedin\.com/in/([a-zA-Z0-9-]+)'
        linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
        if linkedin_match:
            contact_info['linkedin'] = f"linkedin.com/in/{linkedin_match.group(1)}"

        return contact_info

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities using SpaCy.

        Args:
            text: Resume text

        Returns:
            Dict mapping entity type to list of entities
        """
        doc = self.nlp(text)
        entities = defaultdict(list)

        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'PERSON']:
                entities[ent.label_].append(ent.text)

        return dict(entities)

    def parse_resume(self, text: str) -> Dict:
        """
        Parse resume and extract all information.

        Args:
            text: Resume text

        Returns:
            Dict with all parsed information
        """
        return {
            'skills': self.extract_skills(text),
            'all_skills': self.extract_all_skills_flat(text),
            'experience_years': self.extract_experience_years(text),
            'education': self.extract_education(text),
            'contact_info': self.extract_contact_info(text),
            'entities': self.extract_entities(text),
            'text_length': len(text),
            'word_count': len(text.split())
        }

    def compare_skills_with_job(
        self,
        resume_skills: List[str],
        job_requirements: str
    ) -> Dict[str, any]:
        """
        Compare resume skills with job requirements.

        Args:
            resume_skills: List of skills from resume
            job_requirements: Job description text

        Returns:
            Dict with matching analysis
        """
        job_required_skills = self.extract_all_skills_flat(job_requirements)

        matching_skills = set(resume_skills) & set(job_required_skills)
        missing_skills = set(job_required_skills) - set(resume_skills)

        return {
            'matching_skills': sorted(list(matching_skills)),
            'missing_skills': sorted(list(missing_skills)),
            'match_percentage': (
                len(matching_skills) / len(job_required_skills) * 100
                if job_required_skills else 0
            ),
            'total_resume_skills': len(resume_skills),
            'total_required_skills': len(job_required_skills)
        }


# Convenience function
def quick_parse(resume_text: str) -> Dict:
    """
    Quick parse of resume text.

    Args:
        resume_text: Resume text

    Returns:
        Parsed resume dict
    """
    parser = EnhancedResumeParser()
    return parser.parse_resume(resume_text)


if __name__ == "__main__":
    # Demo usage
    print("Enhanced Resume Parser Demo")
    print("=" * 60)

    parser = EnhancedResumeParser()

    sample_resume = """
    John Doe
    john.doe@email.com | (555) 123-4567
    linkedin.com/in/johndoe

    EXPERIENCE
    Senior Data Scientist with 5+ years of experience in machine learning
    and data analysis. Expert in Python, TensorFlow, PyTorch, and AWS.

    EDUCATION
    Master of Science in Computer Science
    Bachelor of Engineering

    SKILLS
    - Programming: Python, R, SQL, JavaScript
    - ML/AI: TensorFlow, PyTorch, scikit-learn, NLP
    - Cloud: AWS, Docker, Kubernetes
    - Data: Pandas, NumPy, Spark, SQL
    """

    result = parser.parse_resume(sample_resume)

    print("\n📄 Parsed Resume:")
    print(f"  Experience: {result['experience_years']} years")
    print(f"  Education: {', '.join(result['education'])}")
    print(f"  Email: {result['contact_info']['email']}")
    print(f"  Total Skills: {len(result['all_skills'])}")
    print(f"\n  Skills by Category:")
    for category, skills in result['skills'].items():
        print(f"    - {category}: {', '.join(skills[:5])}{'...' if len(skills) > 5 else ''}")
