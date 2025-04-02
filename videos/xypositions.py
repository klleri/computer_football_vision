import numpy as np

def bounding_box_center(bbox: list) -> tuple[int, int]:
    """
    Calculates the center coordinates (x, y) of a bounding box.

    Args:
        bbox (list): Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        tuple[int, int]: The (x, y) coordinates of the center.
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

def bounding_box_width(bbox: list) -> int:
    """
    Calculates the width of a bounding box.
    NOTE: Implicitly used in objects.py drawing functions, but not called directly elsewhere.

    Args:
        bbox (list): Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        int: The width (x2 - x1) of the bounding box.
    """
    return int(bbox[2] - bbox[0])

def measure_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Calculates the Euclidean distance between two points.

    Args:
        p1 (tuple[int, int]): Coordinates of the first point (x1, y1).
        p2 (tuple[int, int]): Coordinates of the second point (x2, y2).

    Returns:
        float: The Euclidean distance between p1 and p2.
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def measure_xy_distance(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[int, int]:
    """
    Calculates the difference in x and y coordinates between two points.
    *** NOTE: This function appears to be unused in the current project. ***

    Args:
        p1 (tuple[int, int]): Coordinates of the first point (x1, y1).
        p2 (tuple[int, int]): Coordinates of the second point (x2, y2).

    Returns:
        tuple[int, int]: The difference (p1_x - p2_x, p1_y - p2_y).
    """
    return p1[0] - p2[0], p1[1] - p2[1]

def foot_position(bbox: list) -> tuple[int, int]:
    """
    Estimates the position of the feet based on the bottom-center of the bounding box.
    *** NOTE: This function appears to be unused in the current project. ***

    Args:
        bbox (list): Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        tuple[int, int]: Estimated foot position (center_x, bottom_y).
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    bottom_y = int(y2)
    return center_x, bottom_y