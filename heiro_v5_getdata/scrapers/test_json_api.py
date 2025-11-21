"""
Test the JSON API endpoint to get text listings
"""
import requests
import json

def test_json_api():
    """Test the /json/text/grid endpoint"""
    url = "http://ramses.ulg.ac.be/json/text/grid"
    
    print(f"Testing JSON API: {url}")
    print("="*60)
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'N/A')}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nData type: {type(data)}")
            
            if isinstance(data, list):
                print(f"Number of texts: {len(data)}")
                
                if data:
                    print(f"\nFirst text:")
                    print(json.dumps(data[0], indent=2, ensure_ascii=False))
                    
                    print(f"\nSample of 5 texts:")
                    for i, text in enumerate(data[:5], 1):
                        print(f"{i}. {text.get('displayTitle', 'N/A')} (ID: {text.get('legacyId', 'N/A')})")
                    
                    # Save full data
                    with open('ramses_text_list.json', 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"\n✓ Saved full text list to ramses_text_list.json")
                    
                    return data
            elif isinstance(data, dict):
                print(f"\nData keys: {list(data.keys())}")
                print(json.dumps(data, indent=2, ensure_ascii=False)[:500])
        
        return None
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_text_access(legacy_id):
    """Test accessing a specific text by legacy ID"""
    url = f"http://ramses.ulg.ac.be/text/legacy/{legacy_id}"
    
    print(f"\n{'='*60}")
    print(f"Testing text access: {legacy_id}")
    print(f"URL: {url}")
    print('='*60)
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            # Save sample
            with open(f'sample_text_{legacy_id}.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"✓ Saved to sample_text_{legacy_id}.html")
            
            # Check content length
            print(f"Content length: {len(response.text)} bytes")
            
            return True
        
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    # Test JSON API
    texts = test_json_api()
    
    if texts and len(texts) > 0:
        # Try accessing a few texts
        print(f"\n\n{'='*60}")
        print("Testing text access...")
        print('='*60)
        
        for text in texts[:3]:
            legacy_id = text.get('legacyId')
            if legacy_id:
                test_text_access(legacy_id)

if __name__ == "__main__":
    main()
