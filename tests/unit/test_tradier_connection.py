import requests
import json
from datetime import datetime

def test_tradier_connection():
    # Tradier configuration
    ACCOUNT_ID = "VA69917807"
    TRADIER_TOKEN = "2jacA1MpXvcFWhrg7ZQfUcf4fYww"  # Sandbox token
    ENDPOINT = "https://sandbox.tradier.com/v1"

    # Set up headers
    headers = {
        'Authorization': f'Bearer {TRADIER_TOKEN}',
        'Accept': 'application/json'
    }

    try:
        # Test 1: Get account information
        print("\n1. Testing Account Information:")
        profile_response = requests.get(
            f'{ENDPOINT}/user/profile',
            headers=headers
        )
        profile_response.raise_for_status()
        print(json.dumps(profile_response.json(), indent=2))

        # Test 2: Get account balances
        print("\n2. Testing Account Balances:")
        balance_response = requests.get(
            f'{ENDPOINT}/accounts/{ACCOUNT_ID}/balances',
            headers=headers
        )
        balance_response.raise_for_status()
        print(json.dumps(balance_response.json(), indent=2))

        # Test 3: Get positions
        print("\n3. Testing Current Positions:")
        positions_response = requests.get(
            f'{ENDPOINT}/accounts/{ACCOUNT_ID}/positions',
            headers=headers
        )
        positions_response.raise_for_status()
        print(json.dumps(positions_response.json(), indent=2))

        # Test 4: Get market status
        print("\n4. Testing Market Status:")
        clock_response = requests.get(
            f'{ENDPOINT}/markets/clock',
            headers=headers
        )
        clock_response.raise_for_status()
        print(json.dumps(clock_response.json(), indent=2))

        # Test 5: Get quote for a symbol
        print("\n5. Testing Quote Data:")
        quote_response = requests.get(
            f'{ENDPOINT}/markets/quotes',
            params={'symbols': 'AAPL,NVDA'},
            headers=headers
        )
        quote_response.raise_for_status()
        print(json.dumps(quote_response.json(), indent=2))

        return True

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Tradier: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return False

if __name__ == "__main__":
    print("Starting Tradier Connection Test")
    print(f"Time: {datetime.now()}")
    print("=" * 50)
    
    success = test_tradier_connection()
    
    if success:
        print("\nAll Tradier connection tests completed successfully!")
    else:
        print("\nSome Tradier connection tests failed. Please check the errors above.")