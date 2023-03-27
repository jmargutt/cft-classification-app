import requests

URL = "https://prod-206.westeurope.logic.azure.com:443/workflows/ef9cd65b72754a8c99dfdb3acb0e2b2a/triggers/manual/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=L7QK9UaSktTeBwg0U8gztRBHlVzR76t319yIiQbD5Q0"
body = {
    'email': 'asd@asd',
    'ds_id': 32,
    'model': 'COVID-19 (English)'
}
response = requests.post(URL, json=body)
print(response)