import requests

url = 'http://127.0.0.1:5000/recommend'
data = {
    "handle": "Hamim99"
}
response = requests.post(url, json=data)
print(response.json())
