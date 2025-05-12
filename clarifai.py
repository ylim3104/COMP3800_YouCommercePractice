# from clarifai.client.model import Model
# from clarifai.client.input import ImageInput
#
# # Set up API credentials
# CLARIFAI_PAT = "98969f60a77a4c9189d5cc5c815fffc4"
#
# # Load the pre-trained "General" image model
# model = Model(model_id="general-image-recognition", pat=CLARIFAI_PAT)
#
# # Run prediction on an image (local or URL)
# input = ImageInput(file_path="images/car.jpeg")  # or use `url="https://..."`
#
# # Predict
# result = model.predict(input)
#
# # Output results
# for concept in result.outputs[0].data.concepts:
#     print(f"{concept.name}: {concept.value:.2f}")
