"""
Test the /text search endpoint to find how to list texts
"""
import requests
from bs4 import BeautifulSoup
import json
import time

def search_texts(query=""):
    """Search for texts using the /text endpoint"""
    url = f"http://ramses.ulg.ac.be/text"
    
    print(f"\n{'='*60}")
    print(f"Searching for texts: '{query}'")
    print('='*60)
    
    try:
        # Try GET with query parameter
        params = {'query': query} if query else {}
        response = requests.get(url, params=params, timeout=10)
        
        print(f"Status: {response.status_code}")
        print(f"URL: {response.url}")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for text results
        links = soup.find_all('a', href=True)
        text_links = [a for a in links if '/text/' in a['href'] and a.get_text(strip=True)]
        
        if text_links:
            print(f"\nFound {len(text_links)} text link(s):")
            for link in text_links[:10]:
                href = link['href']
                text = link.get_text(strip=True)
                print(f"  - {text}: {href}")
        
        # Look for forms
        forms = soup.find_all('form')
        if forms:
            print(f"\nFound {len(forms)} form(s):")
            for form in forms:
                action = form.get('action', 'N/A')
                method = form.get('method', 'GET')
                print(f"  Action: {action}, Method: {method}")
                
                inputs = form.find_all(['input', 'select'])
                for inp in inputs:
                    name = inp.get('name', 'N/A')
                    inp_type = inp.get('type', inp.name)
                    print(f"    - {name} ({inp_type})")
        
        # Save for inspection
        with open(f'text_search_{query or "empty"}.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"\nSaved to text_search_{query or 'empty'}.html")
        
        return soup
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def try_known_texts():
    """Try searching for known texts from the corpus page"""
    known_searches = [
        "LESt",  # Late Egyptian Stories abbreviation
        "P. d'Orbiney",  # Document name
        "deux frères",  # Title
        "O. DeM",  # Ostraca from Deir el-Medina
        "LRL",  # Late Ramesside Letters
    ]
    
    for query in known_searches:
        search_texts(query)
        time.sleep(1)  # Be polite

def main():
    # First try empty search to see if it lists all texts
    print("Trying empty search (might list all texts):")
    search_texts("")
    
    time.sleep(1)
    
    # Try known text searches
    print("\n\nTrying known text searches:")
    try_known_texts()

if __name__ == "__main__":
    main()
