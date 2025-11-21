"""
Test accessing Ramses texts via reference endpoint
"""
import requests
from bs4 import BeautifulSoup
import json

def test_reference_endpoint(ref):
    """Try to access a text by reference"""
    url = f"http://ramses.ulg.ac.be/text/gotToReference?ref={ref}"
    print(f"\nTesting reference: {ref}")
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, timeout=10, allow_redirects=True)
        print(f"Status: {response.status_code}")
        print(f"Final URL: {response.url}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for hieroglyphic text content
            # Common patterns: transliteration, translation, metadata
            text_content = soup.get_text()
            
            # Check if we got actual content
            if len(text_content) > 500:
                print(f"✓ Got content ({len(text_content)} chars)")
                
                # Look for specific data elements
                tables = soup.find_all('table')
                if tables:
                    print(f"  Found {len(tables)} table(s)")
                
                # Look for transliteration markers
                if any(marker in text_content.lower() for marker in ['transliteration', 'hieroglyph', 'égyptien']):
                    print("  ✓ Likely contains hieroglyphic data")
                    
                    # Save sample for inspection
                    with open(f'sample_text_{ref.replace("/", "_")}.html', 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    print(f"  Saved sample to sample_text_{ref.replace('/', '_')}.html")
                
                return True
            else:
                print(f"  Content too short, likely error page")
        
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def explore_corpus_page():
    """Check the corpus presentation page for text listings"""
    url = "http://ramses.ulg.ac.be/site/corpus"
    print(f"\n{'='*60}")
    print("Exploring corpus presentation page")
    print('='*60)
    
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find tables
        tables = soup.find_all('table')
        print(f"\nFound {len(tables)} table(s)")
        
        for i, table in enumerate(tables, 1):
            print(f"\nTable {i}:")
            rows = table.find_all('tr')
            print(f"  Rows: {len(rows)}")
            
            # Get sample data
            if rows:
                headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]
                print(f"  Headers/First row: {headers}")
                
                if len(rows) > 1:
                    sample_row = [td.get_text(strip=True) for td in rows[1].find_all('td')]
                    print(f"  Sample data: {sample_row}")
                    
                    # Look for links in the table
                    links = rows[1].find_all('a', href=True)
                    if links:
                        print(f"  Sample link: {links[0]['href']}")
        
        # Save for inspection
        with open('corpus_page.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("\nSaved corpus page to corpus_page.html")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    # First, explore the corpus page to understand structure
    explore_corpus_page()
    
    print(f"\n{'='*60}")
    print("Testing reference endpoints")
    print('='*60)
    
    # Try common reference formats
    # Egyptian text references often use formats like:
    # - KRI (Kitchen, Ramesside Inscriptions)
    # - Papyrus names
    # - Ostraca numbers
    test_refs = [
        "KRI_I_1",
        "KRI_I_1_1",
        "P.BM_10052",
        "O.DeM_1",
        "1",
        "001",
    ]
    
    for ref in test_refs:
        test_reference_endpoint(ref)

if __name__ == "__main__":
    main()
