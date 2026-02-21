def get_center(x1, y1, x2, y2):
    """
    Return center point of a bounding box
    """
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy
