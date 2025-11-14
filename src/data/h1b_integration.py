"""
H1B Visa Sponsorship Data Integration

Integrates DOL (Department of Labor) H1B LCA data to identify
employers that sponsor H1B visas.

Data Sources:
- DOL Labor Condition Application (LCA) disclosure data
- USCIS H-1B Employer Data Hub
"""

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from fuzzywuzzy import fuzz, process

logger = logging.getLogger(__name__)


class H1BSponsorshipData:
    """
    Manager for H1B sponsorship data from DOL and USCIS.
    """

    def __init__(self, cache_dir='h1b_data'):
        """
        Initialize H1B data manager.

        Args:
            cache_dir: Directory to cache downloaded H1B data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.sponsor_data = None
        self.employer_lookup = {}
        self._loaded = False

    def download_dol_data(
        self,
        fiscal_year: int = 2024,
        save_path: Optional[str] = None
    ) -> Path:
        """
        Download DOL H1B LCA disclosure data.

        Note: This is a placeholder. In practice, you would:
        1. Visit: https://www.dol.gov/agencies/eta/foreign-labor/performance
        2. Download the quarterly disclosure data CSV
        3. Save to cache directory

        Args:
            fiscal_year: Fiscal year to download
            save_path: Optional save path

        Returns:
            Path to downloaded file
        """
        if save_path is None:
            save_path = self.cache_dir / f"h1b_lca_fy{fiscal_year}.csv"

        logger.info(f"""
        To download H1B data:
        1. Visit: https://www.dol.gov/agencies/eta/foreign-labor/performance
        2. Download 'H-1B' disclosure data for FY {fiscal_year}
        3. Save to: {save_path}

        Alternatively, use the DOL API:
        - API Sampler: https://devtools.dol.gov/apisampler
        - Endpoint: https://api.dol.gov/v1/lca
        """)

        return Path(save_path)

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load H1B data from local CSV file.

        Expected columns:
        - EMPLOYER_NAME
        - CASE_STATUS (CERTIFIED, DENIED, etc.)
        - PREVAILING_WAGE
        - JOB_TITLE
        - WORKSITE_STATE
        - etc.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with H1B data
        """
        try:
            df = pd.read_csv(filepath, low_memory=False)
            logger.info(f"✓ Loaded {len(df):,} H1B records from {filepath}")

            # Standardize column names
            df.columns = df.columns.str.upper()

            self.sponsor_data = df
            self._build_employer_lookup()
            self._loaded = True

            return df

        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            logger.info("Download H1B data from: https://www.dol.gov/agencies/eta/foreign-labor/performance")
            raise
        except Exception as e:
            logger.error(f"Error loading H1B data: {e}")
            raise

    def _build_employer_lookup(self):
        """Build fast employer lookup with statistics."""
        if self.sponsor_data is None:
            return

        df = self.sponsor_data

        # Normalize employer names
        df['EMPLOYER_NORMALIZED'] = df['EMPLOYER_NAME'].str.upper().str.strip()

        # Calculate statistics per employer
        stats = df.groupby('EMPLOYER_NORMALIZED').agg({
            'CASE_NUMBER': 'count',  # Total applications
            'CASE_STATUS': lambda x: (x == 'CERTIFIED').sum(),  # Certified count
            'PREVAILING_WAGE': 'mean'  # Average wage
        }).reset_index()

        stats.columns = ['employer', 'total_applications', 'certified_count', 'avg_wage']
        stats['approval_rate'] = stats['certified_count'] / stats['total_applications']

        # Convert to dict for fast lookup
        self.employer_lookup = stats.set_index('employer').to_dict('index')

        logger.info(f"✓ Built lookup for {len(self.employer_lookup):,} unique employers")

    def check_employer(
        self,
        employer_name: str,
        fuzzy_match: bool = True,
        threshold: int = 85
    ) -> Optional[Dict]:
        """
        Check if an employer sponsors H1B visas.

        Args:
            employer_name: Employer name to check
            fuzzy_match: Whether to use fuzzy matching
            threshold: Minimum fuzzy match score (0-100)

        Returns:
            Dict with sponsorship info, or None if not found
        """
        if not self._loaded:
            logger.warning("H1B data not loaded. Call load_from_csv() first.")
            return None

        # Normalize input
        employer_normalized = employer_name.upper().strip()

        # Exact match
        if employer_normalized in self.employer_lookup:
            return self._format_employer_info(employer_normalized)

        # Fuzzy match
        if fuzzy_match and self.employer_lookup:
            match = process.extractOne(
                employer_normalized,
                self.employer_lookup.keys(),
                scorer=fuzz.token_sort_ratio
            )

            if match and match[1] >= threshold:
                matched_name, score = match[0], match[1]
                logger.debug(f"Fuzzy matched '{employer_name}' to '{matched_name}' (score: {score})")
                return self._format_employer_info(matched_name, fuzzy_score=score)

        return None

    def _format_employer_info(
        self,
        employer_key: str,
        fuzzy_score: Optional[int] = None
    ) -> Dict:
        """Format employer sponsorship information."""
        info = self.employer_lookup.get(employer_key, {})

        result = {
            'sponsor': True,
            'employer_name': employer_key,
            'total_applications': int(info.get('total_applications', 0)),
            'certified_applications': int(info.get('certified_count', 0)),
            'approval_rate': float(info.get('approval_rate', 0)),
            'avg_h1b_salary': float(info.get('avg_wage', 0)),
        }

        if fuzzy_score is not None:
            result['match_confidence'] = fuzzy_score

        return result

    def enrich_jobs_with_h1b_info(
        self,
        jobs: List[Dict],
        company_field: str = 'job_company',
        fuzzy_match: bool = True
    ) -> List[Dict]:
        """
        Enrich job listings with H1B sponsorship information.

        Args:
            jobs: List of job dictionaries
            company_field: Field name containing company name
            fuzzy_match: Whether to use fuzzy matching

        Returns:
            Jobs list with added 'h1b_sponsor_info' field
        """
        if not self._loaded:
            logger.warning("H1B data not loaded. Returning jobs unchanged.")
            return jobs

        enriched_jobs = []

        for job in jobs:
            job_copy = job.copy()
            company = job.get(company_field, '')

            if company:
                sponsor_info = self.check_employer(company, fuzzy_match=fuzzy_match)
                job_copy['h1b_sponsor_info'] = sponsor_info
            else:
                job_copy['h1b_sponsor_info'] = None

            enriched_jobs.append(job_copy)

        # Count sponsors
        sponsor_count = sum(1 for j in enriched_jobs if j['h1b_sponsor_info'])
        logger.info(f"✓ Found {sponsor_count}/{len(jobs)} jobs from H1B sponsors")

        return enriched_jobs

    def filter_h1b_sponsors(
        self,
        jobs: List[Dict],
        min_approval_rate: float = 0.8,
        min_applications: int = 1
    ) -> List[Dict]:
        """
        Filter jobs to only include H1B sponsors meeting criteria.

        Args:
            jobs: List of jobs (must have h1b_sponsor_info)
            min_approval_rate: Minimum approval rate (0-1)
            min_applications: Minimum number of applications

        Returns:
            Filtered list of jobs
        """
        filtered = []

        for job in jobs:
            info = job.get('h1b_sponsor_info')

            if info is None:
                continue

            if (info.get('approval_rate', 0) >= min_approval_rate and
                info.get('total_applications', 0) >= min_applications):
                filtered.append(job)

        logger.info(f"✓ Filtered to {len(filtered)}/{len(jobs)} jobs from qualified H1B sponsors")

        return filtered

    def get_top_sponsors(self, top_n: int = 100) -> pd.DataFrame:
        """
        Get top H1B sponsors by application volume.

        Args:
            top_n: Number of top sponsors to return

        Returns:
            DataFrame with top sponsors
        """
        if not self._loaded:
            logger.warning("H1B data not loaded.")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(self.employer_lookup, orient='index')
        df = df.sort_values('total_applications', ascending=False).head(top_n)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'employer'}, inplace=True)

        return df

    def generate_sample_data(self, filepath: str):
        """
        Generate sample H1B data for testing (when real data unavailable).

        Args:
            filepath: Path to save sample data
        """
        # Sample data based on known H1B sponsors
        sample_data = {
            'EMPLOYER_NAME': [
                'Amazon.com Services LLC',
                'Microsoft Corporation',
                'Google LLC',
                'Meta Platforms Inc',
                'Apple Inc',
                'IBM Corporation',
                'Intel Corporation',
                'Oracle America Inc',
                'Salesforce Inc',
                'Adobe Inc'
            ] * 100,  # Repeat for sample size
            'CASE_STATUS': ['CERTIFIED'] * 900 + ['DENIED'] * 100,
            'PREVAILING_WAGE': np.random.normal(120000, 30000, 1000),
            'JOB_TITLE': ['Software Engineer', 'Data Scientist', 'ML Engineer'] * 333 + ['Product Manager'],
            'WORKSITE_STATE': ['CA', 'WA', 'NY', 'TX', 'MA'] * 200,
            'CASE_NUMBER': [f'I-200-{i:05d}' for i in range(1000)]
        }

        df = pd.DataFrame(sample_data)
        df.to_csv(filepath, index=False)

        logger.info(f"✓ Generated sample H1B data: {filepath}")
        return df


# Convenience function
def quick_check_sponsor(employer_name: str, data_path: Optional[str] = None) -> Dict:
    """
    Quick check if an employer sponsors H1B.

    Args:
        employer_name: Employer to check
        data_path: Optional path to H1B data CSV

    Returns:
        Sponsorship info dict
    """
    h1b = H1BSponsorshipData()

    if data_path:
        h1b.load_from_csv(data_path)
    else:
        # Try to use sample data
        sample_path = h1b.cache_dir / "sample_h1b_data.csv"
        if not sample_path.exists():
            logger.info("Generating sample H1B data...")
            h1b.generate_sample_data(str(sample_path))
        h1b.load_from_csv(str(sample_path))

    return h1b.check_employer(employer_name)


if __name__ == "__main__":
    # Demo usage
    print("H1B Sponsorship Data Demo")
    print("=" * 60)

    # Initialize
    h1b = H1BSponsorshipData()

    # Generate sample data for demo
    sample_path = h1b.cache_dir / "sample_h1b_data.csv"
    h1b.generate_sample_data(str(sample_path))

    # Load data
    h1b.load_from_csv(str(sample_path))

    # Check some employers
    companies = ["Amazon", "Google", "Microsoft", "Random Startup Inc"]

    print("\nChecking H1B Sponsorship:\n")
    for company in companies:
        info = h1b.check_employer(company)
        if info:
            print(f"✓ {company}")
            print(f"  Total Applications: {info['total_applications']}")
            print(f"  Approval Rate: {info['approval_rate']:.1%}")
            print(f"  Avg Salary: ${info['avg_h1b_salary']:,.0f}")
        else:
            print(f"✗ {company} - No H1B sponsorship data found")
        print()

    # Top sponsors
    print("\nTop 10 H1B Sponsors:")
    print(h1b.get_top_sponsors(10))
