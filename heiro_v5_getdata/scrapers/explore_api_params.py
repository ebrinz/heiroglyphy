import requests

BASE_URL = "http://ramses.ulg.ac.be/json/text/grid"

def test_param(param_name, value):
    url = f"{BASE_URL}?{param_name}={value}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'rows' in data:
                print(f"Param '{param_name}={value}' returned {len(data['rows'])} rows")
                if 'total' in data:
                    print(f"  Total records reported: {data['total']}")
            else:
                print(f"Param '{param_name}={value}' returned JSON without 'rows'")
        else:
            print(f"Param '{param_name}={value}' failed with status {response.status_code}")
    except Exception as e:
        print(f"Param '{param_name}={value}' failed: {e}")

print("Testing API parameters...")
test_param('rows', 20)
test_param('limit', 20)
test_param('page', 2)
test_param('start', 10)  # Sometimes used with limit
