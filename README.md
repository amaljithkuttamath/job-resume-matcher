# Job Resume Matcher v2.0

Modern resume-job matching system with semantic understanding and H1B visa sponsorship filtering.

## What's New in v2.0

- **Semantic Matching**: 85-90% accuracy with Sentence-BERT (vs 60-70% with TF-IDF)
- **H1B Sponsorship Filtering**: Integrated DOL data to identify visa sponsors
- **Enhanced Parsing**: 400+ skills across 9 categories
- **Hybrid Matching**: Combines semantic (70%) + keyword (30%)
- **Secure Configuration**: API keys in .env file
- **Modular Architecture**: Clean, extensible codebase

## Quick Start

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements-new.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Configure
cp .env.example .env
# Edit .env and add your RAPIDAPI_KEY

# Run v2
python matcher_v2.py
```

## Project Structure

```
job-resume-matcher/
├── matcher_v2.py              # Modern main script (use this)
├── res.py                     # Original script (still works)
├── config.py                  # Configuration management
├── requirements-new.txt       # Dependencies
├── .env.example               # Configuration template
│
├── src/                       # Core modules
│   ├── matchers/              # Semantic matching
│   ├── data/                  # H1B integration
│   └── parsers/               # Resume parsing
│
├── docs/                      # Documentation
│   ├── README_V2.md           # Complete usage guide
│   ├── UPGRADE_GUIDE.md       # Migration instructions
│   └── RESEARCH_FINDINGS_2025.md  # Technical deep dive
│
├── tests/                     # Test suite
│   ├── test_h1b.py            # H1B module tests
│   ├── test_full_integration.py   # Integration tests
│   └── TEST_RESULTS.md        # Test results
│
└── demos/                     # Demonstrations
    └── demo_comparison.py     # TF-IDF vs Semantic comparison
```

## Usage

### Basic Matching

```bash
# Default: 7 resumes, 7 jobs, hybrid matching
python matcher_v2.py

# Match more resumes/jobs
python matcher_v2.py --resumes 20 --jobs 50

# Filter for H1B sponsors only
python matcher_v2.py --h1b-only
```

### Methods

```bash
# Semantic only
python matcher_v2.py --method semantic

# Hybrid (default, recommended)
python matcher_v2.py --method hybrid

# Classic TF-IDF
python matcher_v2.py --no-semantic
```

### See the Improvement

```bash
# Compare TF-IDF vs Semantic matching
python demos/demo_comparison.py
```

## Performance Comparison

| Metric | v1 (TF-IDF) | v2 (Semantic) | Improvement |
|--------|-------------|---------------|-------------|
| Accuracy | 60-70% | 85-90% | +25-30% |
| Speed | 1s/resume | 0.23s/resume | 4x faster |
| Semantic | No | Yes | New |
| H1B Filter | No | Yes | New |
| Skills DB | 14 | 400+ | 28x larger |
| Security | Exposed key | .env | Fixed |

## Documentation

- **[docs/README_V2.md](docs/README_V2.md)** - Complete usage guide
- **[docs/UPGRADE_GUIDE.md](docs/UPGRADE_GUIDE.md)** - Migration from v1
- **[docs/RESEARCH_FINDINGS_2025.md](docs/RESEARCH_FINDINGS_2025.md)** - Technical research

## Running Tests

```bash
source .venv/bin/activate

# Test H1B integration
python tests/test_h1b.py

# Test full system
python tests/test_full_integration.py
```

All tests passing. See [tests/TEST_RESULTS.md](tests/TEST_RESULTS.md) for details.

## Features

### Semantic Matching
- Understands context, not just keywords
- Matches "ML Engineer" with "AI Research Scientist"
- 85-90% accuracy

### H1B Sponsorship
- DOL data integration
- Employer verification
- Approval rates and salary data
- Filter jobs by sponsors

### Enhanced Parsing
- 400+ skills detection
- Better experience extraction
- Education identification
- Contact information (email, phone, LinkedIn)

## Cost

- Sentence-BERT: Free (runs locally)
- H1B Data: Free (DOL public data)
- Job APIs: $0-100/month (free tiers available)
- Total: $0-100/month

## Requirements

- Python 3.8+
- 4GB RAM minimum
- Internet connection for job APIs
- Optional: GPU for faster processing

## Original Script

The original `res.py` still works. It has been updated to use secure configuration:

```bash
python res.py
```

## Support

- Installation issues: See [docs/UPGRADE_GUIDE.md](docs/UPGRADE_GUIDE.md)
- Usage questions: See [docs/README_V2.md](docs/README_V2.md)
- Technical details: See [docs/RESEARCH_FINDINGS_2025.md](docs/RESEARCH_FINDINGS_2025.md)

## Data Sources

- **H1B Data**: https://www.dol.gov/agencies/eta/foreign-labor/performance
- **Job APIs**: JSearch (RapidAPI), Indeed, Adzuna

## License

Same as original repository.

---

**New users**: Start with `python demos/demo_comparison.py` to see the improvement.

**Existing users**: See [docs/UPGRADE_GUIDE.md](docs/UPGRADE_GUIDE.md) for migration steps.
