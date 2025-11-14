# Documentation

Complete documentation for the modernized job-resume-matcher v2.0.

## Documentation Files

### README_V2.md - Main Documentation
Complete usage guide for v2.0:
- Quick start guide
- Feature overview
- Usage examples
- Performance comparison
- Programmatic API
- Project structure

**Start here** if you're new to v2.0.

### UPGRADE_GUIDE.md - Migration Guide
Step-by-step guide for upgrading from v1 to v2:
- What's new in v2
- Installation instructions using uv
- Configuration setup
- Migration path
- Usage examples
- FAQ and troubleshooting

**Use this** if you're upgrading from the classic version.

### RESEARCH_FINDINGS_2025.md - Technical Deep Dive
Comprehensive research report covering:
- Current implementation analysis
- Modern AI/ML approaches
- H1B visa sponsorship data integration
- Technology comparison
- Cost analysis
- Implementation roadmap
- Research papers and resources

**Read this** for the complete technical background.

## Quick Navigation

### For New Users
1. Start: `README_V2.md` - Quick start
2. Then: `UPGRADE_GUIDE.md` - Detailed setup
3. Deep dive: `RESEARCH_FINDINGS_2025.md` - Understanding the tech

### For Existing Users
1. Upgrade: `UPGRADE_GUIDE.md` - Migration steps
2. Reference: `README_V2.md` - New features
3. Research: `RESEARCH_FINDINGS_2025.md` - Why these changes

### For Developers
1. Architecture: `RESEARCH_FINDINGS_2025.md` - Design decisions
2. API: `README_V2.md` - Programmatic usage
3. Extend: Look at `src/` modules with inline docs

## Installation with uv

All installation commands in the documentation use `uv` for fast package management:

```bash
# Create virtual environment
uv venv

# Activate
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements-new.txt
```

## Performance Improvements

| Metric | v1 | v2 | Improvement |
|--------|----|----|-------------|
| Accuracy | 60-70% | 85-90% | +25-30% |
| Speed | 1s | 0.23s | 4x faster |
| Semantic | No | Yes | New |
| H1B Filter | No | Yes | New |

## External Resources

- DOL H1B Data: https://www.dol.gov/agencies/eta/foreign-labor/performance
- Sentence Transformers: https://www.sbert.net/
- LangChain: https://python.langchain.com/
- Research Papers: See RESEARCH_FINDINGS_2025.md
