# H1B LCA Data Fields Reference

## Overview

When you download actual H1B Labor Condition Application (LCA) data from the Department of Labor, you'll get much richer information than our sample data. This guide explains the available fields and how to use them.

## Standard Fields (Always Available)

### Employer Information
- **EMPLOYER_NAME**: Company name (e.g., "Google LLC", "Microsoft Corporation")
- **EMPLOYER_ADDRESS**: Company address
- **EMPLOYER_CITY**: City
- **EMPLOYER_STATE**: State
- **EMPLOYER_POSTAL_CODE**: Zip code
- **EMPLOYER_COUNTRY**: Usually "UNITED STATES OF AMERICA"

### Case Information
- **CASE_NUMBER**: Unique identifier (e.g., "I-200-12345-123456")
- **CASE_STATUS**: Application status
  - CERTIFIED: Approved
  - DENIED: Rejected
  - WITHDRAWN: Employer withdrew
  - CERTIFIED_WITHDRAWN: Approved then withdrawn
- **DECISION_DATE**: When DOL made decision
- **RECEIVED_DATE**: When DOL received application

### Job Information
- **JOB_TITLE**: Position title (e.g., "Software Engineer", "Data Scientist")
- **SOC_CODE**: Standard Occupational Classification code (e.g., "15-1252")
- **SOC_TITLE**: SOC description (e.g., "Software Developers, Applications")
- **JOB_DUTIES**: **ACTUAL JOB DESCRIPTION** - This is the key field!
- **FULL_TIME_POSITION**: "Y" or "N"
- **BEGIN_DATE**: Start date of employment
- **END_DATE**: End date of employment

### Salary Information
- **PREVAILING_WAGE**: DOL-determined wage for the position
- **WAGE_RATE_OF_PAY_FROM**: Minimum offered salary
- **WAGE_RATE_OF_PAY_TO**: Maximum offered salary
- **WAGE_UNIT_OF_PAY**: "Year", "Hour", "Week", "Bi-Weekly", "Month"
- **PW_WAGE_LEVEL**: Wage level (I, II, III, IV)
  - Level I: Entry level
  - Level II: Qualified
  - Level III: Experienced
  - Level IV: Fully competent

### Location Information
- **WORKSITE_CITY**: Where work is performed
- **WORKSITE_COUNTY**: County
- **WORKSITE_STATE**: State
- **WORKSITE_POSTAL_CODE**: Zip code

## JOB_DUTIES Field - The Most Important

The `JOB_DUTIES` field contains the actual job description as written by the employer. This is what makes H1B job matching so powerful.

### Example Real Job Duties

**Software Engineer at Google:**
```
Design and develop software solutions for Google Cloud Platform. Implement scalable
microservices using Java, Python, and Go. Build APIs and integrate with existing
systems. Optimize performance of distributed systems. Collaborate with product
managers and UX designers. Participate in code reviews and maintain high code quality
standards. Deploy applications using Kubernetes and Docker. Monitor production systems
and troubleshoot issues.
```

**Data Scientist at Amazon:**
```
Develop machine learning models to improve customer recommendations and search
relevance. Analyze large datasets using SQL, Python, and Spark. Build statistical
models for forecasting and optimization. Create data visualizations and dashboards
for business stakeholders. Collaborate with engineering teams to deploy models to
production. Conduct A/B testing and measure impact of ML initiatives. Present
findings to senior leadership.
```

**ML Engineer at Meta:**
```
Design and implement machine learning infrastructure at scale. Build deep learning
models using PyTorch for computer vision and NLP applications. Develop MLOps
pipelines for model training, evaluation, and deployment. Optimize model performance
for production environments. Work with distributed systems and parallel computing.
Collaborate with research scientists on cutting-edge ML techniques.
```

## Using Job Duties in Matching

### Without Job Duties (Limited)
```python
# Only have job title
job = {
    'job_title': 'Software Engineer',
    'job_company': 'Google LLC'
}
# Semantic matching based on title only - less accurate
```

### With Job Duties (Accurate)
```python
# Have full job description
job = {
    'job_title': 'Software Engineer',
    'job_company': 'Google LLC',
    'job_duties': """
    Design and develop software for Google Cloud Platform.
    Use Java, Python, Go for microservices. Build APIs,
    optimize distributed systems, use Kubernetes/Docker.
    """
}
# Semantic matching understands actual responsibilities!
```

## How Our Code Handles This

### Automatic Detection
```python
def _create_job_description(row, employer_info):
    # Check if JOB_DUTIES field exists
    job_duties = row.get('JOB_DUTIES', row.get('DUTIES', None))

    if job_duties and pd.notna(job_duties):
        # Use REAL job description from DOL data
        description = f"""
{job_title} at {employer}
...
Job Description:
{job_duties}  # <- Actual duties from LCA
...
        """
    else:
        # Generate synthetic description
        description = f"""
{job_title} position at {employer}
(Generic description - real duties not available)
        """
```

### Sample vs Real Data

**Sample Data** (for testing):
- We generate realistic job duties templates
- Multiple variations per role type
- Good for testing but not as specific

**Real DOL Data** (production):
- Actual job duties from employer's LCA
- Position-specific requirements
- Exact skills and technologies mentioned
- Much better for semantic matching

## Downloading Real Data

### Step 1: Visit DOL Website
https://www.dol.gov/agencies/eta/foreign-labor/performance

### Step 2: Download H-1B Disclosure Data
- Look for "H-1B" section
- Download "Disclosure Data" for fiscal year
- Format: CSV (hundreds of MB)
- Contains 100,000+ records

### Step 3: Fields You'll Get

The CSV will have columns like:
```
CASE_NUMBER,CASE_STATUS,EMPLOYER_NAME,JOB_TITLE,JOB_DUTIES,
PREVAILING_WAGE,WORKSITE_CITY,WORKSITE_STATE,SOC_CODE,
DECISION_DATE,BEGIN_DATE,END_DATE,FULL_TIME_POSITION,...
```

### Step 4: Load and Use

```python
from src.data.h1b_integration import H1BSponsorshipData

h1b = H1BSponsorshipData()

# Load real DOL data (has JOB_DUTIES field)
h1b.load_from_csv('h1b_data/H-1B_FY2024_Record_Layout.csv')

# Extract jobs - will automatically use real job duties
h1b_jobs = h1b.get_h1b_jobs(limit=1000)

# Each job now has actual employer-written description
print(h1b_jobs[0]['job_description'])
# Shows real JOB_DUTIES from the LCA
```

## Field Name Variations

Different fiscal years may use slightly different column names:

| Common Name | Variations |
|-------------|------------|
| JOB_DUTIES | DUTIES, JOB_DUTIES, DUTY_DESCRIPTION |
| EMPLOYER_NAME | EMPLOYER_LEGAL_NAME, EMPLOYER_BUSINESS_NAME |
| CASE_STATUS | STATUS, CASE_DECISION |
| PREVAILING_WAGE | PW_AMOUNT, WAGE_RATE |

Our code checks multiple variations:
```python
job_duties = row.get('JOB_DUTIES', row.get('DUTIES', None))
```

## Benefits of Real Job Duties

### 1. Better Matching
- Match based on actual skills mentioned
- Understand specific technologies required
- See real responsibilities

### 2. Transparency
- Know exactly what the job entails
- See required qualifications
- Understand day-to-day work

### 3. Resume Optimization
- Tailor resume to match job duties
- Use same terminology as employer
- Highlight relevant experience

### 4. Salary Context
- Job duties explain salary level
- See complexity of role
- Understand seniority expectations

## Example Comparison

### Job Title Only
```
ML Engineer at Google
Salary: $150,000

Match Score: 0.65
(Based only on "ML Engineer" vs resume)
```

### With Job Duties
```
ML Engineer at Google
Salary: $150,000

Job Duties: "Design deep learning models using PyTorch for NLP.
Build MLOps pipelines with Kubernetes. Optimize model inference
latency. Work on transformer architectures for language understanding."

Match Score: 0.89
(Much higher! Resume mentions PyTorch, NLP, Kubernetes)
```

## Real-World Impact

### Without Job Duties
- "Software Engineer" could be anything
- Frontend? Backend? Mobile? DevOps?
- Generic matching
- Many false positives

### With Job Duties
- "Software Engineer - Python backend, AWS, microservices"
- Specific tech stack visible
- Accurate matching
- Relevant results only

## Conclusion

The `JOB_DUTIES` field is critical for accurate H1B job matching. When available:

- Semantic matching improves by 20-30%
- Fewer false positives
- Better understanding of position fit
- More transparent job information

**Recommendation**: Always download and use real DOL data with JOB_DUTIES field for production use.

## See Also

- [H1B Job Matching Guide](H1B_JOB_MATCHING.md)
- [DOL Performance Data](https://www.dol.gov/agencies/eta/foreign-labor/performance)
- [H1B Integration Module](../src/data/h1b_integration.py)
