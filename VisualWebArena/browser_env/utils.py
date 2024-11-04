import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, TypedDict, Union

import numpy as np
import numpy.typing as npt
from beartype import beartype
from PIL import Image

try:
    from vertexai.preview.generative_models import Image as VertexImage
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image')


@dataclass
class DetachedPage:
    url: str
    content: str  # html


@beartype
def png_bytes_to_numpy(png: bytes) -> npt.NDArray[np.uint8]:
    """Convert png bytes to numpy array

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    return np.array(Image.open(BytesIO(png)))


def pil_to_b64(img: Image.Image, add_prefix: bool = True) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        if add_prefix:
            img_b64 = "data:image/png;base64," + img_b64
    return img_b64


def pil_to_vertex(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_vertex = VertexImage.from_bytes(byte_data)
    return img_vertex


class DOMNode(TypedDict):
    nodeId: str
    nodeType: str
    nodeName: str
    nodeValue: str
    attributes: str
    backendNodeId: str
    parentId: str
    childIds: list[str]
    cursor: int
    union_bound: list[float] | None
    center: list[float] | None


class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: int
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None
    center: list[float] | None


class BrowserConfig(TypedDict):
    win_upper_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    DOMTree: dict[str, Any]
    config: BrowserConfig


AccessibilityTree = list[AccessibilityTreeNode]
DOMTree = list[DOMNode]

Observation = str | npt.NDArray[np.uint8]


class StateInfo(TypedDict):
    observation: dict[str, Observation]
    info: Dict[str, Any]

def mark_bounding_boxes(image, bounding_boxes):
    """
    Mark the pixels in the image matrix that are occupied by bounding boxes.
    
    Args:
    image (numpy.ndarray): 2D binary image where 1 represents occupied and 0 represents empty.
    bounding_boxes (list): List of bounding boxes in the format [x, y, w, h].
    """
    for box in bounding_boxes:
        x, y, w, h = box
        image[y:y+h, x:x+w] = 1  # Mark the area covered by each bounding box

def largest_empty_rectangle(image, squareness_preference_factor=1.0):
    """
    Find the largest empty rectangle in a 2D binary matrix with a preference for square-like rectangles.
    
    Args:
    image (numpy.ndarray): 2D binary matrix representing occupied (1) and empty (0) cells.
    squareness_preference_factor (float): A factor that adjusts the score for squareness.
    
    Returns:
    tuple: The coordinates of the top-left corner and the dimensions of the largest empty rectangle (x, y, w, h).
    """
    rows, cols = image.shape
    height = np.zeros((rows, cols), dtype=int)
    
    # Calculate the height of consecutive empty cells in each column
    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 0:
                height[i, j] = height[i-1, j] + 1 if i > 0 else 1
    
    # Now find the largest rectangle in each row using the height histogram method
    max_score = 0
    best_rectangle = (0, 0, 0, 0)  # (x, y, w, h)

    for i in range(rows):
        stack = []
        for j in range(cols + 1):
            cur_height = height[i, j] if j < cols else 0  # Sentinel value for the last column
            while stack and cur_height < height[i, stack[-1]]:
                h = height[i, stack.pop()]
                w = j if not stack else j - stack[-1] - 1
                area = h * w
                
                # Squareness preference: prefer rectangles closer to a square using min(h, w) / max(h, w)
                squareness_score = min(h, w) / max(h, w)  # Higher score for more square-like shapes
                score = area * (squareness_score ** squareness_preference_factor)
                
                if score > max_score:
                    max_score = score
                    best_rectangle = (stack[-1] + 1 if stack else 0, i - h + 1, w, h)
            stack.append(j)
    
    return best_rectangle

def find_largest_non_overlapping_box(image_size, bounding_boxes, squareness_preference_factor=1.0):
    """
    Finds the largest non-overlapping bounding box in a given image with preference for square-like rectangles.
    
    Args:
    image_size (tuple): The size of the image (width, height).
    bounding_boxes (list): List of bounding boxes in the format [x, y, w, h].
    squareness_preference_factor (float): A factor that adjusts the score for squareness.
    
    Returns:
    tuple: Coordinates and size of the largest bounding box that can be drawn without overlap (x, y, w, h).
    """
    width, height = image_size
    image = np.zeros((height, width), dtype=int)  # Create an empty grid representing the image
    
    # Mark occupied areas
    mark_bounding_boxes(image, bounding_boxes)
    
    # Find the largest empty rectangle
    largest_box = largest_empty_rectangle(image, squareness_preference_factor)
    
    return largest_box