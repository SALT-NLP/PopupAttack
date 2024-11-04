from PIL import Image, ImageDraw, ImageFont
        
def draw_som_for_attack_webarena(
        screenshot_img,
        x,
        y,
        w,
        h,
        tag,
        color,
        bbox_padding=0,
        bbox_border=2,
 ):
    """
    Draw a single bounding box with given coordinates, tag, and color.

    Parameters:
    x, y, w, h: Coordinates and dimensions of the bounding box.
    tag: Text tag to be drawn.
    color: Color of the bounding box and text background.
    bbox_padding: Optional padding around the bounding box.
    bbox_border: Width of the bounding box border.
    """
    # Open the screenshot image
    img = screenshot_img.copy()
    draw = ImageDraw.Draw(img)

    # Load a TTF font with a larger size
    font_path = "media/SourceCodePro-SemiBold.ttf"
    font_size, padding = 16, 2
    font = ImageFont.truetype(font_path, font_size)

    # Draw the bounding box
    left, top, right, bottom = x, y, x + w, y + h
    draw.rectangle(
        [
            left - bbox_padding,
            top - bbox_padding,
            right + bbox_padding,
            bottom + bbox_padding,
        ],
        outline=color,
        width=bbox_border,
    )

    # Calculate the text position
    text_width = draw.textlength(tag, font=font)
    text_height = font_size  # Assume the text is one line
    text_position = (left, top - font_size - padding)
    text_rectangle = [
        text_position[0] - padding,
        text_position[1] - padding,
        text_position[0] + text_width + padding,
        text_position[1] + text_height + padding,
    ]

    # Draw a background rectangle for the text
    draw.rectangle(text_rectangle, fill=color)
    draw.text(text_position, tag, font=font, fill="white")

    return img