import requests
r = requests.get('http://localhost:8000/predict')
print(r)
print(r.json())
