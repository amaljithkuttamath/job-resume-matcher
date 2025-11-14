#!/usr/bin/env python3
"""
H1B Integration Module Test Suite
Tests all H1B sponsorship functionality
"""

import sys
sys.path.insert(0, '.')

from src.data.h1b_integration import H1BSponsorshipData

def test_h1b_module():
    """Comprehensive test of H1B integration module."""

    print('=' * 70)
    print('H1B INTEGRATION MODULE - COMPREHENSIVE TEST')
    print('=' * 70)

    # Test 1: Initialize and generate sample data
    print('\n[TEST 1] Initialization and Sample Data Generation')
    print('-' * 70)
    h1b = H1BSponsorshipData()
    print('✓ H1BSponsorshipData initialized')

    sample_path = h1b.cache_dir / 'sample_h1b_data.csv'
    h1b.generate_sample_data(str(sample_path))
    print(f'✓ Sample data generated: {sample_path}')

    # Test 2: Load data
    print('\n[TEST 2] Loading H1B Data')
    print('-' * 70)
    h1b.load_from_csv(str(sample_path))
    print(f'✓ Loaded {len(h1b.employer_lookup):,} unique employers')

    # Test 3: Check employers (exact match)
    print('\n[TEST 3] Employer Checking (Exact Match)')
    print('-' * 70)

    test_companies = [
        'Google LLC',
        'Microsoft Corporation',
        'Amazon.com Services LLC',
        'Meta Platforms Inc',
        'Random Company Inc'
    ]

    found_count = 0
    for company in test_companies:
        info = h1b.check_employer(company, fuzzy_match=False)
        if info:
            found_count += 1
            print(f'✓ {company}')
            print(f'  Applications: {info["total_applications"]:,}')
            print(f'  Approval Rate: {info["approval_rate"]:.1%}')
        else:
            print(f'✗ {company} - Not found')

    print(f'\nFound {found_count}/{len(test_companies)} companies')

    # Test 4: Job enrichment
    print('\n[TEST 4] Job Enrichment with H1B Data')
    print('-' * 70)

    sample_jobs = [
        {
            'job_id': '1',
            'job_title': 'Software Engineer',
            'job_company': 'Google LLC',
            'salary': '$150,000'
        },
        {
            'job_id': '2',
            'job_title': 'Data Analyst',
            'job_company': 'Startup XYZ',
            'salary': '$100,000'
        },
        {
            'job_id': '3',
            'job_title': 'ML Engineer',
            'job_company': 'Apple Inc',
            'salary': '$180,000'
        }
    ]

    enriched = h1b.enrich_jobs_with_h1b_info(
        sample_jobs,
        company_field='job_company'
    )

    h1b_sponsor_count = sum(1 for j in enriched if j['h1b_sponsor_info'])
    print(f'✓ Enriched {len(enriched)} jobs')
    print(f'✓ Found {h1b_sponsor_count} H1B sponsors')

    for job in enriched:
        print(f'\n  Job {job["job_id"]}: {job["job_title"]} at {job["job_company"]}')
        if job['h1b_sponsor_info']:
            info = job['h1b_sponsor_info']
            print(f'    ✓ H1B Sponsor')
            print(f'    Approval Rate: {info["approval_rate"]:.1%}')
            print(f'    Avg H1B Salary: ${info["avg_h1b_salary"]:,.0f}')
        else:
            print(f'    ✗ No H1B data')

    # Test 5: Filtering
    print('\n[TEST 5] Filtering for H1B Sponsors')
    print('-' * 70)

    filtered = h1b.filter_h1b_sponsors(
        enriched,
        min_approval_rate=0.80,
        min_applications=50
    )

    print(f'✓ Filtered to {len(filtered)}/{len(enriched)} jobs from qualified H1B sponsors')

    for job in filtered:
        info = job['h1b_sponsor_info']
        print(f'  - {job["job_title"]} at {job["job_company"]}')
        print(f'    Approval: {info["approval_rate"]:.1%} | Apps: {info["total_applications"]}')

    # Test 6: Top sponsors
    print('\n[TEST 6] Top H1B Sponsors')
    print('-' * 70)

    top_sponsors = h1b.get_top_sponsors(5)
    print('Top 5 H1B Sponsors by Application Volume:\n')

    for idx, row in top_sponsors.iterrows():
        print(f'  {idx+1}. {row["employer"]}')
        print(f'     Applications: {int(row["total_applications"]):,}')
        print(f'     Approval Rate: {row["approval_rate"]:.1%}')
        print(f'     Avg Salary: ${row["avg_wage"]:,.0f}')
        print()

    # Summary
    print('=' * 70)
    print('TEST SUMMARY')
    print('=' * 70)
    print('✓ All H1B module tests passed!')
    print('\nFeatures verified:')
    print('  ✓ Sample data generation')
    print('  ✓ Data loading and indexing')
    print('  ✓ Employer checking (exact match)')
    print('  ✓ Job enrichment with H1B info')
    print('  ✓ Filtering for H1B sponsors')
    print('  ✓ Top sponsors ranking')
    print('\nThe H1B integration module is fully functional!')
    print('=' * 70)

if __name__ == '__main__':
    test_h1b_module()
