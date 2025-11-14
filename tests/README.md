# Tests

Comprehensive test suite for the modernized job-resume-matcher v2.0.

## Test Files

### test_h1b.py - H1B Integration Module Tests
Tests all H1B sponsorship functionality:
- Sample data generation
- Employer checking (exact and fuzzy match)
- Job enrichment with H1B info
- Filtering for visa sponsors
- Top sponsors ranking

**Run**: `python3 tests/test_h1b.py`

### test_full_integration.py - Complete System Integration
Tests the complete workflow with semantic matching + H1B filtering:
- Semantic matching with Sentence-BERT
- H1B sponsorship data enrichment
- Combined filtering and ranking
- Smart job recommendations

**Run**: `python3 tests/test_full_integration.py`

### TEST_RESULTS.md - Detailed Test Results
Comprehensive documentation of all test results, including:
- Performance metrics
- Comparison tables
- Sample outputs
- Known limitations
- Production readiness assessment

## Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run H1B tests
python3 tests/test_h1b.py

# Run full integration test
python3 tests/test_full_integration.py
```

## Test Results Summary

All tests passing:
- H1B integration: 100% functional
- Semantic matching: 85-90% accuracy
- Full integration: Production ready

See `TEST_RESULTS.md` for detailed analysis.
