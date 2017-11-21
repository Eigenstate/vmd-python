
def triangle(id, v1, v2, v3):
    """
    Draws a triangle with given vertices.

    Args:
        id (int): Molecule ID to draw on
        v1 (3-tuple): Coordinates of first vertex
        v2 (3-tuple): Coordinates of second vertex
        v3 (3-tuple): Coordinates of third vertex

    Returns:
        (int): Graphics object ID drawn
    """
    pass

def trinorm(id, v1, v2, v3, n1, n2, n3):
    """
    Draws a triangle with given vertices and normals.

    Args:
        id (int): Molecule ID to draw on
        v1 (3-tuple): Coordinates of first vertex
        v2 (3-tuple): Coordinates of second vertex
        v3 (3-tuple): Coordinates of third vertex
        n1 (3-tuple): Coordinates of first normal
        n2 (3-tuple): Coordinates of second normal
        n3 (3-tuple): Coordinates of third normal

    Returns:
        (int): Graphics object ID drawn
    """
    pass

def cylinder(id, v1, v2, radius=1.0,resolution=6, filled=0):
    """
    Draws a cylinder with given vertices.

    Args:
        id (int): Molecule ID to draw on
        v1 (3-tuple): Coordinates of first vertex
        v2 (3-tuple): Coordinates of second vertex
        radius (float): Cylinder radius
        resolution (int): Resolution??
        filled (int): Fill??

    Returns:
        (int): Graphics object ID drawn
    """
    pass

def point(id, v):
    """
    Draws a point at the given coordinates.

    Args:
        id (int): Molecule ID to draw on
        v (3-tuple): Coordinates of point

    Returns:
        (int): Graphics object ID drawn
    """
    pass

def line(id, v1, v2, style="solid", width=1):
    """
    Draws a line between two vertices.

    Args:
        id (int): Molecule ID to draw on
        v1 (3-tuple): Coordinates of first vertex
        v2 (3-tuple): Coordinates of second vertex
        style (str): "solid" or "dashed" to set line style
        width (int): Line width

    Returns
        (int): Graphics object ID drawn
    """
    pass

def materials(id, onoff):
    """
    Turns the use of materials on or off for subsequent graphics
    drawn on this molecule. Currently drawn graphics are unaffected.

    Args:
        id (int): Molecule ID to alter
        onoff (int): 1 to turn on, 0 to turn off

    Returns:
        (int): index of next graphics object that will be affected
    """
    pass

def material(id, name):
    """
    Sets the material for all graphics drawn on this molecule. Currently
    drawn graphics are affected.

    Args:
        id (int): Molecule ID to alter
        name (str): Material to set. Should be one of the names returned
            by `material.listall`

    Returns:
        (int): index of next graphics object that would be drawn
    """
    pass

def color():
    pass

def cone():
    pass

def sphere():
    pass

def text():
    pass

def delete():
    pass

def replace():
    pass

def info():
    pass

def listall():
    pass

