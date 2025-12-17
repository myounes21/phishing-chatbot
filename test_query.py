#!/usr/bin/env python3
"""
Test script to verify the query endpoint is working
Run this while the server is running to diagnose issues
"""

import requests
import json
import sys

API_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_query():
    """Test the query endpoint"""
    print("\n" + "=" * 60)
    print("Testing Query Endpoint")
    print("=" * 60)
    try:
        payload = {
            "query": "Hello, are you working?",
            "include_sources": True
        }
        response = requests.post(
            f"{API_URL}/query_agent",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            print(f"Response: {data.get('response', '')[:200]}...")
            print(f"Sources: {len(data.get('sources', []))} sources")
        else:
            print(f"❌ Error Response:")
            try:
                error_data = response.json()
                print(f"Detail: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"Response Text: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("\n🔍 Testing Phishing Chatbot API\n")
    
    # Test health
    health_ok = test_health()
    
    # Test query
    query_ok = test_query()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Health Endpoint: {'✅ OK' if health_ok else '❌ FAILED'}")
    print(f"Query Endpoint: {'✅ OK' if query_ok else '❌ FAILED'}")
    
    if not health_ok:
        print("\n⚠️ Health endpoint failed. Check if server is running:")
        print(f"   uvicorn main:app --reload")
    
    if health_ok and not query_ok:
        print("\n⚠️ Health check passed but query failed.")
        print("   Check server logs for detailed error messages.")
        print("   The error detail above should tell you what's wrong.")
    
    sys.exit(0 if (health_ok and query_ok) else 1)
