# Demos

Interactive demonstrations of the modernized job-resume-matcher features.

## Demo Files

### demo_comparison.py - TF-IDF vs Semantic Matching

**What it shows**:
- Side-by-side comparison of classic TF-IDF vs modern Sentence-BERT matching
- Real-world examples with different job types
- Performance differences
- Why semantic understanding matters

**Key demonstration**:
- TF-IDF: Only matches exact keywords
- Semantic: Understands "ML Engineer" is similar to "AI Research Scientist"
- Hybrid: Best of both worlds

**Run**: `python3 demos/demo_comparison.py`

**Expected output**:
```
Quick Demo:
- Shows how semantic matching understands synonyms
- TF-IDF misses "programmer" = "engineer"
- Semantic correctly identifies similarity

Full Comparison:
- 5 job postings with varying relevance
- Rankings by each method
- Accuracy analysis
```

## Running Demos

```bash
# Activate environment
source .venv/bin/activate

# Run comparison demo
python3 demos/demo_comparison.py
```

## What You'll Learn

1. **Semantic Advantage**: Why modern embeddings beat keyword matching
2. **Real Impact**: 25-30% accuracy improvement
3. **Practical Use**: How to apply these techniques
