import requests

# Replace with your DeepAI API Key
API_KEY = '86a6f64f-4a28-4327-b1c6-857c91e82edb'

# Local image file path
image_path = 'images/car_1.jpeg'  # Make sure this image exists in your directory

# Prepare API request
response = requests.post(
    "https://api.deepai.org/api/object-detection",
    files={'image': open(image_path, 'rb')},
    headers={'api-key': API_KEY}
)

# Print response JSON
print(response.json())
