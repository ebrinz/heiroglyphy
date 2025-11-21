"""
Base Scraper Class for Ancient Egyptian Text Sources

Provides common functionality for all scrapers:
- Rate limiting
- Error handling and retry logic
- Data validation
- Progress tracking
- Output formatting
"""

import time
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import json
from pathlib import Path
from tqdm import tqdm


class BaseScraper(ABC):
    """Abstract base class for all hieroglyphic text scrapers"""
    
    def __init__(self, output_dir: str = "data/raw", rate_limit: float = 1.0):
        """
        Args:
            output_dir: Directory to save raw scraped data
            rate_limit: Minimum seconds between requests
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Heiroglyphy Academic Research Bot (erik.brinsmead@gmail.com)'
        })
    
    def _rate_limit_wait(self):
        """Ensure we respect rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _safe_request(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """Make a request with retry logic"""
        for attempt in range(max_retries):
            try:
                self._rate_limit_wait()
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Failed to fetch {url}: {e}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        return None
    
    @abstractmethod
    def scrape(self) -> List[Dict]:
        """
        Main scraping method to be implemented by subclasses
        
        Returns:
            List of dictionaries with keys:
                - transliteration: str
                - translation: str (if available)
                - source: str
                - metadata: dict (period, text_type, etc.)
        """
        pass
    
    def save_results(self, data: List[Dict], filename: str):
        """Save scraped data to JSON file"""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} records to {output_path}")
