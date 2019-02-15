"""
Contains methods for controlling how trajectories are displayed in the
view window. All active molecules will be affected.
"""

def once():
    """
    Animates once through all frames
    """
    pass

def rock():
    """
    Animates back and forth between the first and last frames
    """
    pass

def loop():
    """
    Plays trajectory in a continuous loop
    """
    pass

def style():
    """
    Obtains current animation style

    Returns:
        (str): Current animation style, in 'rock', "once', or 'loop'
    """
    pass

def forward():
    """
    Plays the current trajectory forwards
    """
    pass

def reverse():
    """
    Plays the current trajectory in reverse
    """
    pass

def prev():
    """
    Animates to the previous frame, then stops
    """
    pass

def next():
    """
    Animates to the next frame, then stops
    """
    pass

def pause():
    """
    Pauses the animation
    """
    pass

def speed(value):
    """
    Sets the current animation speed

    Args:
        value (float): The speed to set the animation to, or -1 to get the
            current speed

    Returns:
        (float): The new speed
    """
    pass

def skip(value):
    """
    Sets the animation stride

    Args:
        value (int): The number of frames to skip each animation step, or -1
            to get the current stride

    Returns:
        (int): The new stride
    """
    pass

def is_active(molid):
    """
    Query if a molecule is active / animateable

    Args:
        molid (int): Molecule ID to query

    Returns:
        (bool): If molecule is active

    Raises:
        ValueError: If molecule ID is invalid
    """
    pass

def activate(molid, activate):
    """
    Set the active status of the given molecule.

    Args:
        molid (int): Molecule ID to set
        activate (bool): The new active status of the molecule

    Raises:
        ValueError: If molecule ID is invalid
    """
    pass
