"""
Exploration script to understand Ramses Online structure
"""
import requests
from bs4 import BeautifulSoup
import json

def explore_endpoint(url, description):
    """Fetch and analyze an endpoint"""
    print(f"\n{'='*60}")
    print(f"Exploring: {description}")
    print(f"URL: {url}")
    print('='*60)
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for forms (search interfaces)
        forms = soup.find_all('form')
        if forms:
            print(f"\nFound {len(forms)} form(s):")
            for i, form in enumerate(forms, 1):
                print(f"\n  Form {i}:")
                print(f"    Action: {form.get('action', 'N/A')}")
                print(f"    Method: {form.get('method', 'N/A')}")
                inputs = form.find_all(['input', 'select', 'textarea'])
                if inputs:
                    print(f"    Inputs:")
                    for inp in inputs:
                        name = inp.get('name', 'N/A')
                        inp_type = inp.get('type', inp.name)
                        print(f"      - {name} ({inp_type})")
        
        # Look for links to texts
        text_links = soup.find_all('a', href=True)
        text_urls = [a['href'] for a in text_links if '/text/' in a['href']]
        if text_urls:
            print(f"\nFound {len(text_urls)} text-related links")
            print("Sample URLs:")
            for url in list(set(text_urls))[:5]:
                print(f"  - {url}")
        
        # Look for data tables
        tables = soup.find_all('table')
        if tables:
            print(f"\nFound {len(tables)} table(s)")
        
        # Look for JSON/API endpoints in scripts
        scripts = soup.find_all('script')
        api_mentions = []
        for script in scripts:
            if script.string and ('api' in script.string.lower() or 'json' in script.string.lower()):
                api_mentions.append(script.string[:200])
        
        if api_mentions:
            print(f"\nFound {len(api_mentions)} script(s) mentioning API/JSON")
        
        return soup
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    base_url = "http://ramses.ulg.ac.be"
    
    # Explore key endpoints
    endpoints = [
        (f"{base_url}/", "Homepage"),
        (f"{base_url}/text", "Text search"),
        (f"{base_url}/text/viewReference", "View reference"),
        (f"{base_url}/search/simple", "Simple search"),
        (f"{base_url}/site/corpus", "Corpus presentation"),
    ]
    
    for url, desc in endpoints:
        explore_endpoint(url, desc)
    
    # Try to find if there's a text listing or API
    print("\n" + "="*60)
    print("Attempting to find text listings...")
    print("="*60)
    
    # Common patterns for text databases
    test_urls = [
        f"{base_url}/text/list",
        f"{base_url}/text/browse",
        f"{base_url}/api/texts",
        f"{base_url}/texts",
    ]
    
    for url in test_urls:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"\n✓ Found: {url}")
                print(f"  Content-Type: {response.headers.get('content-type', 'N/A')}")
                print(f"  Length: {len(response.content)} bytes")
        except:
            pass

if __name__ == "__main__":
    main()
