from pydantic import BaseModel
from grok_setup import *
import cv2
from IPython.display import Image, display
import random

lions_image = f"{IMAGES_DIR}/lions.png"
Image(lions_image, width=600)
print(
    analyze_image(
    lions_image, "How many lions are in this picture? Are they male, female, cubs?"
    )
)

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
        object_detection_prompt: str,
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
                        "url": f"data:image/jpg;base64,{base64_encode_image(image_path)}",
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
    box_thickness: int = 8,
    label_size: float = 1.0,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image file: {image_path}")

    height, width, _ = image.shape

    for box in bounding_boxes.boxes:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        start_point = (int(box.x1 * width), int(box.y1 * height))
        end_point = (int(box.x2 * width), int(box.y2 * height))

        image = cv2.rectangle(image, start_point, end_point, color, box_thickness)
        label = box.object_name

        text_thickness = max(1, int(label_size * 1.5))
        label_dimensions, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, label_size, text_thickness
        )
        label_position = (start_point[0], start_point[1] - 20)

        # Adjust the background rectangle dynamically based on label dimensions
        cv2.rectangle(
            image,
            (label_position[0], label_position[1] - label_dimensions[1] - 6),
            (label_position[0] + label_dimensions[0], label_position[1] + 6),
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

    display(Image(data=cv2.imencode(".jpg", image)[1].tobytes(), width=600))

def detect_objects(
    grok_client: OpenAI,
    image_path: str,
    user_prompt: str,
    object_detection_prompt: str,
) -> None:
    bounding_boxes = generate_bounding_boxes(
        grok_client, image_path, user_prompt, object_detection_prompt
    )
    draw_bounding_boxes(image_path, bounding_boxes)

object_detection_prompt = """
You are an AI assistant specialized in object detection and drawing accurate bounding boxes. Your task is to generate normalized coordinates for bounding boxes based on given instructions and an image.


The coordinates for the bounding boxes should be normalized relative to the width and height of the image. This means:
- The top-left corner of the image is (0, 0)
- The bottom-right corner of the image is (1, 1)
- X-coordinates increase from left to right
- Y-coordinates increase from top to bottom

Now, here are the specific instructions for object detection:
<instructions>
{USER_INSTRUCTIONS}
</instructions>

Your output should be in the following format:
<bounding_box>
  <object>Name of the object</object>
  <coordinates>
    <top_left>(x1, y1)</top_left>
    <bottom_right>(x2, y2)</bottom_right>
  </coordinates>
</bounding_box>

If there are multiple objects to detect, provide separate <bounding_box> entries for each object. Focus on identifying the coordinates for one object before moving on to the next.

Here's an example of a good output:
<bounding_box>
  <object>Cat</object>
  <coordinates>
    <top_left>(0.2, 0.3)</top_left>
    <bottom_right>(0.5, 0.7)</bottom_right>
  </coordinates>
</bounding_box>
<bounding_box>
  <object>Dog</object>
  <coordinates>
    <top_left>(0.6, 0.4)</top_left>
    <bottom_right>(0.9, 0.8)</bottom_right>
  </coordinates>
</bounding_box>

If you cannot detect an object mentioned in the instructions, or if the instructions are unclear, include an explanation in your response:
<error>Unable to detect [object name] because [reason]</error>

Remember to be as accurate as possible when determining the coordinates. If you're unsure about the exact position of an object, use your best judgment to provide the most reasonable estimate.

Begin your object detection and bounding box coordinate generation now, based on the provided image and instructions.
"""

# detect_objects(
#     grok_client,
#     lions_image,
#     user_prompt="Detect all the lions, lionesses and cubs in this image",
#     object_detection_prompt=object_detection_prompt,
# )
