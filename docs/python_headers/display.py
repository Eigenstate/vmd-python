def update():
    """
    Forces a render window update. Does not update the GUI or TUI.
    """
    pass

def update_ui():
    """
    Updates the render window and all user interfaces
    """
    pass

def update_on():
    """
    Regularly updates the render window.
    If the frame is changed, the molecule is rotated etc, the window
    will frequently update to reflect this.
    """
    pass

def update_off():
    """
    Tells VMD to not update the render window.
    If the frame is changed, the molecule is rotated etc, the window
    will not reflect these changes.
    """
    pass

def set(**kwargs):
    """
    Sets a display parameter. This function can accomplish the same
    functionality as the GUI Display->Display Options menu. Any option
    or combination of options may be specified.

    Args:
        eyesep (float): Distance between eyes for stereo display
        focallength (float): Focal length
        height (float): Screen height
        distance (float): Screen distance
        nearclip (float): Near clip plane
        farclip (float): Far clip plane
        antialias (bool): If anti-aliasing is used
        depthcueue (bool): If depth cueing should be done
        culling (bool): If surface culling should be done
        ambientocclusion (bool): If ambient occlusion should used
        stereo (string): Stereo mode, in output from stereomodes()
        projection (string): Projection mode, either 'Persepective'
            or 'Orthographic'
        size (2-tuple of int): Render window size
        aoambient (float): Ambient lighting for ambient occlusion
        aodirect (float): Direct lighting for ambient occlusion
        shadows (bool): If shadows should be rendered
        dof (bool): If rendering should be done with depth of field
        dof_fnumber (float): F-number for depth of field
        dof_focaldist (float): Focal distance for depth of field

    Raises:
        ValueError: If an invalid value or attribute was given
    """
    pass

def get(parameter):
    """
    Returns the given display parameter.

    Args:
        parameter (str): Parameter to query. Must be an allowable argument
            to set()

    Returns:
       (int, float, 2-tuple, or bool) Requested attribute, of appropriate type

    Raises:
        ValueError: If an invalid attribute was specified
        ValueError: If an internal error occurred retrieving the parameter
    """
    pass

def stereomodes():
    """
    Get allowed stereo modes for this display device

    Returns:
        (list of str): Supported stereo modes
    """
    pass

