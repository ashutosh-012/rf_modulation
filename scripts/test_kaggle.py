import urllib.request
import os

token = "KGAT_ed4455c39af97f5dda184fefd8727259"
url = "https://www.kaggle.com/api/v1/datasets/download/nolasthitnotomorrow/radioml2016-deepsigcom"

req = urllib.request.Request(url)
req.add_header("Authorization", f"Bearer {token}")

try:
    print("Testing Kaggle Bearer Token...")
    # Just do a HEAD request or open it and read a few bytes to see if auth works
    response = urllib.request.urlopen(req)
    print(f"Success! Response code: {response.getcode()}")
    print(f"File size: {response.headers.get('Content-Length')} bytes")
except Exception as e:
    print(f"Failed: {e}")
