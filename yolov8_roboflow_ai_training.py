import json

from roboflow import Roboflow

rf = Roboflow(api_key="5Ja5xXj52WjI4G0rd76h")
# project = rf.workspace("ai-training-dzyte").project("car-damage-detection-dzxim-1in4q")
project = rf.workspace("ai-training-dzyte").project("etiquetado-de-danos-vxeta")
# print(project.versions())
model = project.version(1).model
# model = project.version(2).model

INPUT = 'images/car_7.jpeg'
OUTPUT = 'result/prediction3_car7.jpeg'
result = model.predict(INPUT, confidence=40, overlap=30).json()
model.predict(INPUT, confidence=40, overlap=30).save(OUTPUT)
with open('result/output.txt', 'a+') as output:
    output.write(f'{result}\n')
print(result)