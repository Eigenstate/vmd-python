
def categories():
    """
    Get available color categories.

    A few example color categories are "Name", "Type", "Surface", and "Display"

    Returns: (list) Available color categories. 
    """
    pass

def get_colormap(name):
    """
    Get name/color pairs in the given category

    Args:
        name (str): A color category, from categories()

    Returns:
        (dict str->str): Item name, color name in category
    """
    pass

def set_colormap(name, pairs):
    """
    Set name/color pairs in the given category

    Args:
        name (str): A color category, from categories()
        pairs (dict str-> str): New item name, color name pair(s) to update

    Raises:
        ValueError: Invalid color category or item specified
    """
    pass

def get_colors():
    """
    Get color name to RGB correspondences.

    Returns:
        (dict str->3-tuple of floats): Color name, RGB tuple of all defined colors

    Raises:
        ValueError: If a color definition cannot be retrieved
    """
    pass

def get_colorlist():
    """
    Get all colors in RGB

    Returns:
        (list of 3-tuple of floats): RGB values of all defined colors

    Raises:
        ValueError: If a color definition cannot be retrieved
    """
    pass

def set_colors(values):
    """
    Sets color name to RGB correspondences.

    Args:
        (dict str->3-tuple of floats): Color name, new RGB value

    Raises:
        ValueError: If color name undefined
        ValueError: If color definition is not a 3-tuple of floats
    """
    pass

def set_colorid(id, color):
    """
    Sets given color index to a new RGB value

    Args:
        id (int): Color ID to set
        rgb (3-tuple of floats): New RGB value for color

    Raises:
        ValueError: If color index is out of range
    """
    pass

def scale_method():
    """
    Get current colorscale method

    Returns:
        (str): Name of current colorscale method
    """
    pass

def scale_methods():
    """
    Get all valid colorscale methods

    Returns:
        (list of str): Available colorscale methods
    """
    pass

def scale_midpoint():
    """
    Get current colorscale midpoint value

    Returns:
        (float): Midpoint value of current colorscale
    """
    pass

def scale_min():
    """
    Get current colorscale minimum value

    Returns:
        (float): Minimum value of current colorscale
    """
    pass

def scale_max():
    """
    Get current colorscale maximum value

    Returns:
        (float): Maximum value of current colorscale
    """
    pass

def set_scale(method=None, midpoint=None, min=None, max=None):
    """
    Set colorscale parameters.

    Optional arguments are handled correctly if all arguments
    are passed as keywords.

    Args:
        method (str): Colorscale method to set, or None for current scale
        midpoint (float): Midpoint color value to set
        min (float): Minimum color value to set
        max (float): Maximum color value to set

    Raises:
        ValueError: If color scale method is invalid
    """
    pass

