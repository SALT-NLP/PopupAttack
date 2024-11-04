import io
import xml.etree.ElementTree as ET
from typing import Tuple, List
import copy
from PIL import Image, ImageDraw, ImageFont


def find_leaf_nodes(xlm_file_str):
    if not xlm_file_str:
        return []

    root = ET.fromstring(xlm_file_str)

    # Recursive function to traverse the XML tree and collect leaf nodes
    def collect_leaf_nodes(node, leaf_nodes):
        # If the node has no children, it is a leaf node, add it to the list
        if not list(node):
            leaf_nodes.append(node)
        # If the node has children, recurse on each child
        for child in node:
            collect_leaf_nodes(child, leaf_nodes)

    # List to hold all leaf nodes
    leaf_nodes = []
    collect_leaf_nodes(root, leaf_nodes)
    return leaf_nodes


state_ns = "uri:deskat:state.at-spi.gnome.org"
component_ns = "uri:deskat:component.at-spi.gnome.org"


def judge_node(node: ET, platform="ubuntu", check_image=False) -> bool:
    keeps: bool = node.tag.startswith("document") \
                  or node.tag.endswith("item") \
                  or node.tag.endswith("button") \
                  or node.tag.endswith("heading") \
                  or node.tag.endswith("label") \
                  or node.tag.endswith("scrollbar") \
                  or node.tag.endswith("searchbox") \
                  or node.tag.endswith("textbox") \
                  or node.tag.endswith("link") \
                  or node.tag.endswith("tabelement") \
                  or node.tag.endswith("textfield") \
                  or node.tag.endswith("textarea") \
                  or node.tag.endswith("menu") \
                  or node.tag in {"alert", "canvas", "check-box"
                      , "combo-box", "entry", "icon"
                      , "image", "paragraph", "scroll-bar"
                      , "section", "slider", "static"
                      , "table-cell", "terminal", "text"
                      , "netuiribbontab", "start", "trayclockwclass"
                      , "traydummysearchcontrol", "uiimage", "uiproperty"
                      , "uiribboncommandbar"
                                  }
    keeps = keeps and (platform == "ubuntu" \
                       and node.get("{{{:}}}showing".format(state_ns), "false") == "true" \
                       and node.get("{{{:}}}visible".format(state_ns), "false") == "true" \
                       or platform == "windows" \
                       and node.get("{{{:}}}visible".format(state_ns), "false") == "true" \
                       ) \
            and (node.get("{{{:}}}enabled".format(state_ns), "false") == "true" \
                 or node.get("{{{:}}}editable".format(state_ns), "false") == "true" \
                 or node.get("{{{:}}}expandable".format(state_ns), "false") == "true" \
                 or node.get("{{{:}}}checkable".format(state_ns), "false") == "true"
                 ) \
            and (node.get("name", "") != "" or node.text is not None and len(node.text) > 0 \
                 or check_image and node.get("image", "false") == "true"
                 )

    coordinates: Tuple[int, int] = eval(node.get("{{{:}}}screencoord".format(component_ns), "(-1, -1)"))
    sizes: Tuple[int, int] = eval(node.get("{{{:}}}size".format(component_ns), "(-1, -1)"))
    keeps = keeps and coordinates[0] >= 0 and coordinates[1] >= 0 and sizes[0] > 0 and sizes[1] > 0
    return keeps


def filter_nodes(root: ET, platform="ubuntu", check_image=False):
    filtered_nodes = []

    for node in root.iter():
        if judge_node(node, platform, check_image):
            filtered_nodes.append(node)
            # print(ET.tostring(node, encoding="unicode"))

    return filtered_nodes


import random
import io
from PIL import Image, ImageDraw, ImageFont

def draw_bounding_boxes(nodes, image_file_content, down_sampling_ratio=1.0, attack=""):
    # Load the screenshot image
    image_stream = io.BytesIO(image_file_content)
    image = Image.open(image_stream)
    if float(down_sampling_ratio) != 1.0:
        image = image.resize((int(image.size[0] * down_sampling_ratio), int(image.size[1] * down_sampling_ratio)))

    if attack != "":
        # Determine the total number of nodes and select a random reserved index
        total_nodes = 0
        for _node in nodes:
            coords_str = _node.attrib.get('{uri:deskat:component.at-spi.gnome.org}screencoord')
            size_str = _node.attrib.get('{uri:deskat:component.at-spi.gnome.org}size')

            if coords_str and size_str:
                total_nodes += 1
        
        if total_nodes == 0:
            reserved_index = -1
        else:
            reserved_index = random.randint(1, min(total_nodes, 300))
    else:
        # no reservation if no attack
        reserved_index = -1
    
    # keep the original one
    copied_image = copy.deepcopy(image)

    while True:
        image = copy.deepcopy(copied_image)

        draw = ImageDraw.Draw(image)
        marks = []
        drew_nodes = []
        text_informations: List[str] = ["index\ttag\tname\ttext"]

        try:
            # Adjust the path to the font file you have or use a default one
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            # Fallback to a basic font if the specified font can't be loaded
            font = ImageFont.load_default()

        # Loop over all the visible nodes and draw their bounding boxes
        index = 1
        for _node in nodes:
            coords_str = _node.attrib.get('{uri:deskat:component.at-spi.gnome.org}screencoord')
            size_str = _node.attrib.get('{uri:deskat:component.at-spi.gnome.org}size')

            if coords_str and size_str:
                try:
                    # Parse the coordinates and size from the strings
                    coords = tuple(map(int, coords_str.strip('()').split(', ')))
                    size = tuple(map(int, size_str.strip('()').split(', ')))

                    original_coords = copy.deepcopy(coords)
                    original_size = copy.deepcopy(size)

                    if float(down_sampling_ratio) != 1.0:
                        # Downsample the coordinates and size
                        coords = tuple(int(coord * down_sampling_ratio) for coord in coords)
                        size = tuple(int(s * down_sampling_ratio) for s in size)

                    # Check for negative sizes
                    if size[0] <= 0 or size[1] <= 0:
                        raise ValueError(f"Size must be positive, got: {size}")

                    # Calculate the bottom-right corner of the bounding box
                    bottom_right = (coords[0] + size[0], coords[1] + size[1])

                    # Check that bottom_right > coords (x1 >= x0, y1 >= y0)
                    if bottom_right[0] < coords[0] or bottom_right[1] < coords[1]:
                        raise ValueError(f"Invalid coordinates or size, coords: {coords}, size: {size}")

                    # change the index for the reserved index
                    if index == reserved_index:
                        index += 1

                    # Check if the area only contains one color
                    cropped_image = image.crop((*coords, *bottom_right))
                    if len(set(list(cropped_image.getdata()))) == 1:
                        continue

                    # Draw rectangle on image
                    draw.rectangle([coords, bottom_right], outline="red", width=1)

                    # Draw index number at the bottom left of the bounding box with black background
                    text_position = (coords[0], bottom_right[1])  # Adjust Y to be above the bottom right
                    text_bbox: Tuple[int, int, int, int] = draw.textbbox(text_position, str(index), font=font, anchor="lb")
                    draw.rectangle(text_bbox, fill='black')
                    draw.text(text_position, str(index), font=font, anchor="lb", fill="white")

                    # each mark is an x, y, w, h tuple
                    marks.append([original_coords[0], original_coords[1], original_size[0], original_size[1]])
                    drew_nodes.append(_node)

                    if _node.text:
                        node_text = (_node.text if '"' not in _node.text \
                                        else '"{:}"'.format(_node.text.replace('"', '""'))
                                    )
                    elif _node.get("{uri:deskat:uia.windows.microsoft.org}class", "").endswith("EditWrapper") \
                            and _node.get("{uri:deskat:value.at-spi.gnome.org}value"):
                        node_text: str = _node.get("{uri:deskat:value.at-spi.gnome.org}value")
                        node_text = (node_text if '"' not in node_text \
                                        else '"{:}"'.format(node_text.replace('"', '""'))
                                    )
                    else:
                        node_text = '""'
                    text_information: str = "{:d}\t{:}\t{:}\t{:}" \
                        .format(index, _node.tag
                                , _node.get("name", "")
                                , node_text
                                )
                    text_informations.append(text_information)

                    index += 1

                except ValueError:
                    pass

        if index <= reserved_index:
            if index < 1:
                break
            # Just redo once
            reserved_index = random.randint(1, index)
        else:
            break

    output_image_stream = io.BytesIO()
    image.save(output_image_stream, format='PNG')
    image_content = output_image_stream.getvalue()

    return marks, drew_nodes, "\n".join(text_informations), image_content, reserved_index


def print_nodes_with_indent(nodes, indent=0):
    for node in nodes:
        print(' ' * indent, node.tag, node.attrib)
        print_nodes_with_indent(node, indent + 2)
