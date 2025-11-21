import requests
import json

def fetch_sample():
    # 1. Get the list
    print("Fetching list...")
    resp = requests.get("http://ramses.ulg.ac.be/json/text/grid")
    data = resp.json()
    
    if 'rows' in data and data['rows']:
        first_text = data['rows'][0]
        legacy_id = first_text.get('legacyId')
        print(f"Found text with legacyId: {legacy_id}")
        
        # 2. Get the text page
        url = f"http://ramses.ulg.ac.be/text/legacy/{legacy_id}"
        print(f"Fetching {url}...")
        text_resp = requests.get(url)
        
        filename = f"sample_ramses_{legacy_id}.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text_resp.text)
        print(f"Saved to {filename}")
    else:
        print("No rows found in JSON response")

if __name__ == "__main__":
    fetch_sample()
