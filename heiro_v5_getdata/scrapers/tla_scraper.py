"""
Enhanced TLA Scraper

Expands beyond the HuggingFace dataset by directly scraping
the TLA website for additional texts and metadata.
"""

from base_scraper import BaseScraper
from typing import List, Dict
from datasets import load_dataset


class TLAScraper(BaseScraper):
    """Enhanced scraper for Thesaurus Linguae Aegyptiae"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_name = "TLA"
    
    def scrape_huggingface(self) -> List[Dict]:
        """Load existing HuggingFace dataset"""
        print("Loading TLA dataset from HuggingFace...")
        dataset = load_dataset(
            "thesaurus-linguae-aegyptiae/tla-Earlier_Egyptian_original-v18-premium",
            split="train"
        )
        
        results = []
        for item in dataset:
            results.append({
                'transliteration': item.get('transliteration', ''),
                'translation': item.get('translation_de', ''),  # German
                'source': 'TLA (HuggingFace)',
                'metadata': {
                    'text_id': item.get('id', ''),
                    'period': 'Earlier Egyptian'
                }
            })
        
        print(f"Loaded {len(results)} texts from HuggingFace")
        return results
    
    def scrape(self) -> List[Dict]:
        """
        Scrape TLA data
        
        Currently just uses HuggingFace dataset.
        Can be extended to scrape additional data from TLA website.
        """
        return self.scrape_huggingface()


if __name__ == "__main__":
    scraper = TLAScraper()
    data = scraper.scrape()
    scraper.save_results(data, "tla_raw.json")
