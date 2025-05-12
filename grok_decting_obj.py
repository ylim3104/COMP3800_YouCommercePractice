import os
from dotenv import load_dotenv
import base64
from openai import OpenAI
import mimetypes
from pydantic import BaseModel
import cv2
from IPython.display import Image, display
import random

load_dotenv()

XAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.x.ai/v1"
IMAGES_DIR = "images"
GROK_VISION_MODEL = "grok-2-vision-latest"
GROK_MODEL = "grok-2-latest"
grok_client = OpenAI(base_url=BASE_URL, api_key=XAI_API_KEY)
text_file_prompt = """ Generate a detailed, structured, and human-readable report summarizing the following for the provided car image, formatted as a markdown table:
Damaged Parts: Identify and list all specific parts of the car that are visibly damaged (e.g., front bumper, headlight, hood, fender). Do not use 'left', 'right', 'driver's side', or 'passenger's side' in labels.
Type of Damage: Describe the nature of the damage for each part (e.g., dent, crack, scratch, severe deformation).
Estimated Repair Time: Provide an estimated time in hours for repairing each damaged part, based on typical automotive repair standards (e.g., 2-4 hours for a dented hood).
Estimated Cost: Provide an estimated cost in Canadian Dollars (CAD) for repairing or replacing each part, based on 2025 Canadian automotive repair market rates (e.g., $200-$500 for a headlight replacement).
Recommended Action: Suggest whether to repair or replace each damaged part, considering the extent of damage and cost-effectiveness (e.g., repair minor dents, replace cracked headlights).
If a part cannot be accurately assessed, include: <error>Unable to assess [part_name] because [reason]</error>

Additional Instructions:
If the image is unclear, low-resolution, or only shows part of the car, note which areas are not visible (e.g., "Rear of the car not visible") and avoid speculative analysis.
If visual cues are insufficient to assess damage severity, use general automotive repair knowledge to provide reasonable estimates.
If no damage is visible, state: "No visible damage detected in the provided image."
Ensure all estimates are grounded in typical repair practices and market rates for Canada in 2025.
Strictly avoid using 'left', 'right', 'driver's side', or 'passenger's side' in any part labels to maintain consistency and clarity.

Output Format: Provide the report as a markdown table with columns for Damaged Parts, Type of Damage, Estimated Repair Time, Estimated Cost, and Recommended Action. Include any errors or notes in a separate section below the table.

Example Output:
| Damaged Parts    | Type of Damage | Estimated Repair Time | Estimated Cost (CAD) | Recommended Action |
|------------------|----------------|-----------------------|---------------------|-------------------|
| Front Bumper     | Scratch        | 1-2 hours             | $150-$300           | Repair            |
| Headlight        | Crack          | 1-2 hours             | $200-$500           | Replace           |
| Hood             | Dent           | 2-4 hours             | $300-$600           | Repair            |

**Notes**:
- `<error>Unable to assess rear bumper because rear of the car is not visible</error>`

"""

object_detection_prompt = """
You are an AI assistant specialized in object detection and drawing accurate bounding boxes. Your task is to identify and label all damaged areas in the provided car image, generating normalized coordinates for bounding boxes based on the analysis.

The coordinates for the bounding boxes should be normalized relative to the width and height of the image:
- The top-left corner of the image is (0, 0)
- The bottom-right corner of the image is (1, 1)
- X-coordinates increase from left to right
- Y-coordinates increase from top to bottom

**Instructions**:
- Detect all visible damaged areas on the car.
- Label each damaged area with a specific description (e.g., 'Damaged Front Bumper', 'Cracked Headlight', 'Dented Hood'). Do not use 'left', 'right', 'driver's side', or 'passenger's side' in labels.
- Provide separate <bounding_box> entries for each distinct damaged area, even if areas overlap.
- If no damage is detected, output: <message>No visible damage detected in the provided image</message>
- If an area of damage is unclear or undetectable, include: <error>Unable to detect [description] because [reason]</error>
- For ambiguous boundaries, use conservative estimates for the bounding box, prioritizing the most visible edges of the damaged area.
- If the image is low-resolution, partially obscured, or only shows part of the car, note which areas are not visible (e.g., 'Rear of the car not visible') and avoid speculative detection.
- When rendering bounding boxes and labels visually, use a smaller yet clearly visible font size and textbox (e.g., reduce font to 70% of default size, ensure contrast for readability). Adjust line thickness of bounding boxes to be thinner but distinct (e.g., 1-2 pixels).

**Output Format**:
<bounding_box>
  <object>Name of the damaged area</object>
  <coordinates>
    <top_left>(x1, y1)</top_left>
    <bottom_right>(x2, y2)</bottom_right>
  </coordinates>
</bounding_box>

**Example Output**:
<bounding_box>
  <object>Damaged Front Bumper</object>
  <coordinates>
    <top_left>(0.3, 0.6)</top_left>
    <bottom_right>(0.5, 0.8)</bottom_right>
  </coordinates>
</bounding_box>
<bounding_box>
  <object>Cracked Headlight</object>
  <coordinates>
    <top_left>(0.55, 0.4)</top_left>
    <bottom_right>(0.65, 0.5)</bottom_right>
  </coordinates>
</bounding_box>
<error>Unable to detect rear damage because rear of the car is not visible</error>
"""



def base64_encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

def analyze_image(image_path: str, prompt: str, output_folder: str, base_filename: str):
    mime_type = get_mime_type(image_path)
    encoded_image = base64_encode_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_image}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    completion = grok_client.chat.completions.create(
        model = GROK_VISION_MODEL, messages=messages
    )

    output = completion.choices[0].message.content

    with open(f'{output_folder}/xai_output_{base_filename}.txt', 'w+') as file:
        file.write(output)
    print(output)


def get_mime_type(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "image/jpeg"

class BoundingBox(BaseModel):
    object_name: str
    y1: float
    x1: float
    y2: float
    x2: float

class BoundingBoxes(BaseModel):
    boxes: list[BoundingBox]

def generate_bounding_boxes(
        client: OpenAI,
        image_path: str,
        user_query: str,
) -> BoundingBoxes:
    prompt_with_instructions = object_detection_prompt.format(
        USER_INSTRUCTIONS=user_query
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{get_mime_type(image_path)};base64,{base64_encode_image(image_path)}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": prompt_with_instructions},
            ],
        },
    ]

    completion = client.chat.completions.create(
        model=GROK_VISION_MODEL, messages=messages
    )

    if not completion.choices[0].message.content:
        raise ValueError("Expected message content on response was not found")

    semi_structured_response = completion.choices[0].message.content

    completion = client.beta.chat.completions.parse(
        model=GROK_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"What are the coordinates of all bounding boxes: {semi_structured_response}",
            }
        ],
        response_format=BoundingBoxes,
    )

    bounding_boxes = completion.choices[0].message.parsed
    if not bounding_boxes:
        raise ValueError("No bounding boxes extracted")

    return bounding_boxes



def draw_bounding_boxes(
    image_path: str,
    bounding_boxes: BoundingBoxes,
    output_folder: str,
    base_filename: str
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image file: {image_path}")

    height, width, _ = image.shape

    scale_factor = min(width, height) / 1000
    box_thickness = max(1, int(1 * scale_factor))
    label_size = max(0.2, 0.4 * scale_factor)
    text_thickness = max(1, int(label_size * 1.5))
    padding = max(3, int(5 * scale_factor))

    occupied_y_positions = []

    for box in bounding_boxes.boxes:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        shrink_factor = 0.1
        x1 = box.x1 + shrink_factor * (box.x2 - box.x1)
        x2 = box.x2 - shrink_factor * (box.x2 - box.x1)
        y1 = box.y1 + shrink_factor * (box.y2 - box.y1)
        y2 = box.y2 - shrink_factor * (box.y2 - box.y1)

        start_point = (int(x1 * width), int(y1 * height))
        end_point = (int(x2 * width), int(y2 * height))

        image = cv2.rectangle(image, start_point, end_point, color, box_thickness)
        label = box.object_name

        label_dimensions, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, label_size, text_thickness
        )

        label_y = start_point[1] - label_dimensions[1] - padding
        label_x = start_point[0]

        for occupied_y in occupied_y_positions:
            if abs(label_y - occupied_y) < label_dimensions[1] + padding:
                label_y = occupied_y - label_dimensions[1] - padding
                break

        label_y = max(label_dimensions[1] + padding, label_y)
        label_position = (label_x, label_y)

        occupied_y_positions.append(label_y)

        cv2.rectangle(
            image,
            (label_position[0], label_position[1] - label_dimensions[1] - 4),
            (label_position[0] + label_dimensions[0], label_position[1] + 4),
            color,
            -1,
        )

        cv2.putText(
            image,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            label_size,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )

    display(Image(data=cv2.imencode(".jpeg", image)[1].tobytes(), width=600))
    cv2.imwrite(f"{output_folder}/{base_filename}_boxed.jpeg", image)

def detect_objects(
    image_path: str,
    user_prompt: str,
    output_folder: str,
    base_filename: str

) -> None:
    bounding_boxes = generate_bounding_boxes(
        grok_client, image_path, user_prompt
    )
    draw_bounding_boxes(image_path, bounding_boxes, output_folder, base_filename)


def generate_xai(image_path: str, output_folder: str):
    Image(image_path, width=600)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    analyze_image(image_path, text_file_prompt, output_folder, base_filename)
    detect_objects(
        image_path,
        "Detect all the areas of damages in this car image",
        output_folder,
        base_filename
    )

def main():
   pass