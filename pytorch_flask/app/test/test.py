import requests

resp = requests.post("http://localhost:5000/predict", files={'file': open('seven.png','rb')}) #sending request to the server
#and opens the file in read binary mode 

print(resp.text) #this will print the json data returned in main.py
