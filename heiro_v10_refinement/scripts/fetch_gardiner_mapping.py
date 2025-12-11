import requests
import json
import re
from bs4 import BeautifulSoup
from pathlib import Path

def fetch_gardiner_mapping():
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": "List_of_Egyptian_hieroglyphs",
        "format": "json",
        "prop": "text"
    }
    
    headers = {
        "User-Agent": "HeiroglyphyBot/1.0 (crashy@example.com)"
    }
    
    print(f"Fetching {params['page']} from Wikipedia API...")
    response = requests.get(url, params=params, headers=headers)
    
    try:
        data = response.json()
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Response text:")
        print(response.text[:500])  # Print first 500 chars
        return None
    
    if "error" in data:
        print(f"Error: {data['error']}")
        return None
        
    html_content = data["parse"]["text"]["*"]
    soup = BeautifulSoup(html_content, "html.parser")
    
    mapping = {}
    
    # Find all tables
    tables = soup.find_all("table", class_="wikitable")
    print(f"Found {len(tables)} tables.")
    
    if tables:
        print("HTML of the first table (first 1000 chars):")
        print(str(tables[0])[:1000])
    
    for table in tables:
        # Check headers to see if this is a sign list table
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        
        # Typical headers: "code", "image", "unicode", "transliteration", "description", "notes"
        # We need at least "code" (or similar) and "transliteration"
        
        # Sometimes headers are in the first row of tr if th is missing
        if not headers:
             first_row = table.find("tr")
             if first_row:
                 headers = [td.get_text(strip=True).lower() for td in first_row.find_all(["th", "td"])]
        
        print(f"Table {len(mapping)} headers: {headers}") # Debug print
        
        # Identify column indices
        code_idx = -1
        trans_idx = -1
        
        # Check if first header is a template artifact (e.g. 'vtelist...')
        offset = 0
        if headers and ("vte" in headers[0] or "list of" in headers[0]):
            offset = 1
            print("Detected header offset. Shifting indices by -1.")

        for i, h in enumerate(headers):
            if "gardiner" in h or "code" in h:
                code_idx = i - offset
            if "phonogram" in h or "transliteration" in h or "value" in h:
                trans_idx = i - offset
        
        print(f"Table {len(mapping)}: code_idx={code_idx}, trans_idx={trans_idx}")
                
        if code_idx != -1 and trans_idx != -1:
            # Iterate rows
            rows = table.find_all("tr")[1:] # Skip header
            
            # Debug: print first 5 rows HTML
            if len(mapping) == 0:
                print("HTML of first 5 rows:")
                for r in rows[:5]:
                    print(str(r)[:500])
            
            for row in rows:
                cols = row.find_all(["td", "th"])
                if len(cols) > max(code_idx, trans_idx):
                    # Extract code
                    code_text = cols[code_idx].get_text(strip=True)
                    
                    # Remove U+ and everything after it
                    if "U+" in code_text:
                        code_text = code_text.split("U+")[0]
                    
                    # Regex to find the Gardiner code (Letter + Number + optional letter)
                    # e.g. A1, A1a, D21, Aa1
                    # Must start with A-Z or Aa, followed by number
                    match = re.search(r'([A-Za-z]+[0-9]+[A-Za-z]*)', code_text)
                    
                    if match:
                        code = match.group(1)
                        
                        # Extract transliteration
                        trans = cols[trans_idx].get_text(strip=True)
                        
                        # Clean up transliteration
                        trans = re.sub(r'\[.*?\]', '', trans).strip()
                        
                        # Only add if we have a valid code and transliteration
                        if code and trans:
                            mapping[code] = trans
                    else:
                        # Debug: print skipped rows to ensure we aren't missing data
                        # print(f"Skipping row: {code_text}")
                        pass
                        
    print(f"Extracted {len(mapping)} mappings.")
    return mapping

if __name__ == "__main__":
    mapping = fetch_gardiner_mapping()
    
    if mapping:
        # Output to heiro_v10_refinement/data
        output_dir = Path("heiro_v10_refinement/data")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "gardiner_mapping.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        print(f"Saved mapping to {output_path.absolute()}")
        
        # Print sample
        print("\nSample entries:")
        for k in list(mapping.keys())[:10]:
            print(f"{k}: {mapping[k]}")
