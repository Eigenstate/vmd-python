
def evaltcl(command):
    """
    Evaluates a command with the TCL interpreter.
    This allows all VMD functionality that does not yet have Python
    bindings to be used. The TCL interpreter state persists between
    calls to evaltcl, and variables set and modules imported will be
    available on subsequent calls to this function.

    Args:
        command (str): TCL command to be run.

    Returns:
        (str): Output from TCL intepreter

    Raises:
        ValueError: Any error that TCL throws
    """
