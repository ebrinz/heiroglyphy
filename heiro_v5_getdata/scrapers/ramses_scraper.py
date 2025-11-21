"""Ramses Online Scraper

Scrapes hieroglyphic texts from the Ramses Online database.
URL: http://ramses.ulg.ac.be/

This implementation uses the internal JSON API to retrieve the text list
and parses the legacy text view HTML to extract transliterations and translations.
"""

from base_scraper import BaseScraper
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import time


class RamsesScraper(BaseScraper):
    """Scraper for Ramses Online database"""
    
    BASE_URL = "http://ramses.ulg.ac.be/"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_name = "Ramses Online"
    
    def scrape(self, max_texts: Optional[int] = None) -> List[Dict]:
        """
        Scrape hieroglyphic texts from Ramses Online
        
        Args:
            max_texts: Optional limit on number of texts to process
        """
        print(f"Scraping from {self.BASE_URL}...")
        results = []
        
        # Step 1: Get list of texts from JSON API
        # Default to a large number if no limit specified to ensure we get all texts
        limit = max_texts if max_texts else 10000
        list_url = f"{self.BASE_URL}json/text/grid?rows={limit}"
        response = self._safe_request(list_url)
        if not response:
            print("Failed to fetch text list")
            return results
            
        try:
            data = response.json()
            if isinstance(data, dict) and 'rows' in data:
                texts = data['rows']
            else:
                print("Unexpected JSON format for text list")
                return results
        except ValueError:
            print("Failed to parse JSON response")
            return results
            
        print(f"Found {len(texts)} texts available")
        
        # Apply limit if requested
        if max_texts:
            texts = texts[:max_texts]
            print(f"Limiting to first {max_texts} texts")
        
        # Step 2: Process each text
        for i, text_info in enumerate(texts):
            legacy_id = text_info.get('legacyId')
            title = text_info.get('displayTitle')
            
            if not legacy_id:
                continue
                
            print(f"Processing text {i+1}/{len(texts)}: {title} (ID: {legacy_id})")
            
            text_data = self._process_text(legacy_id, text_info)
            if text_data:
                results.extend(text_data)
                
            # Be polite
            self._rate_limit_wait()
            
        return results

    def _process_text(self, legacy_id: int, text_info: Dict) -> List[Dict]:
        """Process a single text, handling pagination"""
        blocks = []
        page = 1
        
        while True:
            url = f"{self.BASE_URL}text/legacy/{legacy_id}?page={page}"
            print(f"  Fetching page {page}: {url}")
            response = self._safe_request(url)
            if not response:
                print("  Failed to fetch page")
                break
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text blocks from this page
            page_blocks = self._extract_blocks(soup, text_info)
            if not page_blocks:
                # If page has no content, we've likely reached the end
                # even if pagination buttons exist
                break
                
            blocks.extend(page_blocks)
            
            # Check for next page
            # Look for a 'next' link in pagination that isn't disabled
            pagination = soup.find('ul', class_='paginationBar')
            if not pagination:
                break
                
            next_li = pagination.find('li', class_='next')
            if not next_li or 'disabled' in next_li.get('class', []):
                break
                
            # If we found a valid next button, increment page
            page += 1
            self._rate_limit_wait()
            
        return blocks

    def _extract_blocks(self, soup: BeautifulSoup, text_info: Dict) -> List[Dict]:
        """Extract text blocks from a parsed page"""
        blocks = []
        
        # Find all glossed text sections
        glossed_divs = soup.find_all('div', class_='glossedText')
        
        for div in glossed_divs:
            # 1. Extract Transliteration
            # There might be multiple tables (lines) per block
            transliteration_parts = []
            
            tables = div.find_all('table', class_='ramsesLine')
            for table in tables:
                tr_trans = table.find('tr', class_='transliteration')
                if tr_trans:
                    # Find all transliteration divs inside words
                    words = tr_trans.find_all('div', class_='transliteration')
                    line_trans = [w.get_text(strip=True) for w in words]
                    transliteration_parts.extend(line_trans)
            
            transliteration = " ".join(transliteration_parts)
            
            # 2. Extract Translation
            translation = ""
            p_trans = div.find('p', class_='translation')
            if p_trans:
                translation = p_trans.get_text(strip=True)
            
            if transliteration:
                blocks.append({
                    "transliteration": transliteration,
                    "translation": translation,
                    "source": "Ramses Online",
                    "metadata": {
                        "text_id": str(text_info.get('legacyId')),
                        "title": text_info.get('displayTitle'),
                        "dating": text_info.get('dating', {}).get('label'),
                        "text_type": text_info.get('textType', {}).get('label'),
                        "period": "Late Egyptian"
                    }
                })
                
        return blocks


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape Ramses Online")
    parser.add_argument("--limit", type=int, help="Limit number of texts to scrape")
    args = parser.parse_args()
    
    scraper = RamsesScraper()
    data = scraper.scrape(max_texts=args.limit)
    
    if data:
        scraper.save_results(data, "ramses_raw.json")
