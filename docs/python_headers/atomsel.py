"""
Methods for manipulating selections of atoms
"""

def get_molecule(selection):
    """
    Get the molecule belonging to a selection.

    Args:
        selection (atomsel): Atom selection to query

    Returns:
        (int): Molecule ID that the selection is on

    Raises:
        ValueError: if molecule has been deleted or is invalid
    """
    pass

class atomsel(object):
    """
    An atom selection object

    Attributes:
        molid (int): Molecule ID selection applies to
        frame (int): Frame selection applies to
        bonds (list of list of int): Atom indices each atom in selection
            is bonded to, inorder
    """

    def __init__(self, selection, molid=None, frame=None):
        """
        Create a new atom selection object

        Args:
            selection (str): VMD syntax for atoms to select
            molid (int): Molecule to select on, defaults to top molecule
            frame (int): Frame to select on, defaults to current frame.

        Raises:
            ValueError: if molecule ID is invalid
            ValueError: if atom selection text cannot be parsed
        """
        pass

    def get(self, attribute):
        """
        Returns the given attribute for each atom in the selection.
        Not all attributes are always valid, depending on which data
        are present in the loaded molecule.

        Args:
            attribute (str): Desired attribute. See attributes list

        Returns:
            (list): Attribute for each atom in selection, in order

        Raises:
            ValueError: If requested attribute is invalid
        """
        pass

    def set(self, attribute, values):
        """
        Sets the given attribute for each atom in the selection. If a single
        value is passed, sets the entire selection to have that value.
        Otherwise, pass a list of the same length as the selection to set
        the attribute for each atom in the selection, inorder.

        Args:
            attribute (str): Desired attribute (see list)
            values (dtype or list of dtype): Values to set. Data type depends
                on the attribute.

        Raises:
            ValueError: if attribute is invalid or not modifiable
            ValueError: if passed list of values is the wrong length
        """
        pass

    def getframe(self):
        """
        Obtains the frame the selection currently refers to.

        Returns:
            (int): The frame number
        """
        pass

    def setframe(self, frame):
        """
        Sets the frame the selection refers to. This does not necessarily
        update the atoms selected by the command. Use update() to do that.

        Args:
            frame (int): The frame number. Set to -1 for the current
                frame, or -2 for the last possible frame

        Raises:
            ValueError: if frame is invalid for selection's molecule ID
        """
        pass

    def update(self):
        """
        Recomputes atoms in selection. If the selection is distance
        based or otherwise depends on atom locations, changes to the
        frame will not be reflected in the selection
        """
        pass

    def write(self, filetype, filename):
        """
        Writes the atoms in the selection to a file.

        Args:
            filetype (str): File format to write
            filename (str): Filename to write to

        Raises:
            ValueError: if molecule cannot be saved
        """
        pass

    def minmax(self):
        """
        Returns the minimum and maximum coordinates in the selection.
        X, Y, and Z axes are treated independently.

        Returns:
            (list of list): Minimum, then maximum X,Y, and Z coordinates.
        """
        pass

    def centerperresidue(self, weight=None):
        """
        Obtain the center of each residue in the selection, with optional
        weighting.

        Args:
            weight (list of float): Per-atom weighting to apply. Must be
                the same length as the selection.

        Returns:
            (list of list): Each element is the X,Y,Z coordinates of
                the center of each residue in the selection

        Raises:
            ValueError: if weight is the wrong size as selection
            ValueError: if non floating point values are in weight
        """
        pass


    def center(self, weight=None):
        """
        Obtains the center of the selection, with optional weighting.

        Args:
            weight (list of float): Per-atom weighting to apply. Must
                be the same length as the selection.

        Returns:
            (tuple): The X,Y,Z coordinate of the center of the selection
        """
        pass

    def rmsd(self, selection, weight=None):
        """
        Obtains the root-mean-square distance between two selections with
        optional weighting. Selections must have the same number of atoms.
        If provided, weight must be the same size as well.

        Args:
            selection (atomsel): The atom selection to compare to. Must be
                same length as this selection.
            weight (list of float): Per-atom weights, must be same length
                as the selection.

        Returns:
            (float): RMSD between selections

        Raises:
            ValueError: if selections have differing numbers of atoms
            ValueError: if weight is the wrong size or contains non-floats
        """
        pass

    def rmsdQCP(self, selection, weight=None):
        """
        Obtains the root-mean-square distance between two selections following
        optimal rotation, with optional weights. Selections must have the same
        number of atoms. If provided, weight must be the same length as well.

        Args:
            selection (atomsel): The atomsel to compare to. Must be the same
                length as this selection
            weight (list of float): Per-atom weights, must be same length as
                this selection.

        Returns:
            (float): RMSD between selections

        Raises:
            ValueError: if selections have differing number of atoms
            ValueError: if weight is the wrong size or contains non-floats
        """
        pass

    def rmsf(self, first=0, last=-1, step=1):
        """
        Measures the root-mean-square-fluctuation along a trajectory.
        By default, goes over all frames.

        Args:
            first (int): First trajectory frame to consider
            last (int): Last trajectory frame to consider, or -1 for
                last available
            step (int): Step size between frames

        Returns:
            (list of float): Root-mean-square-fluctuation of each atom
                in the selection

        Raises:
            RuntimeError: if RMSF could not be calculated
        """
        pass

    def rgyr(self, selection=None, weight=None):
        """
        Obtains the radius of gyration of a selection, with optional
        weighting. Weight must be None of a list of the same length
        as selection.

        Args:
            selection (atomsel): ?????
            weight (list of float): Per-atom weighting. Must be same
                length as this selection

        Returns:
            (float): Radius of gyration of the selection

        Raises:
            ValueError: if calculation could not be completed
        """
        pass

    def rmsfperresidue(self, first=0, last=-1, step=1):
        """
        Measures the root-mean-square-fluctuation along a trajectory
        on a per-residue basis. By default, goes over all frames.

        Args:
            first (int): The index of the first frame to include
            last (int): The index of the last frame to include,
                or -1 for the last frame in the trajectory
            step (int): The step size when iterating through all frames

        Returns:
            (list of float): The RMSF for each residue in the selection

        Raises:
            RuntimeError: if RMSF calculation fails somehow
        """
        pass

    def rmsdperresidue(self, selection, weight=None):
        """
        Obtains the root-mean-square distance between two selections
        on a per-residue basis, with optional weighting. Both selections
        must have the same number of atoms. If provided,
        the weights need to be the same length as the selection.

        Args:
            selection (atomsel): The atom selection to compare to. Must be
                same length as this selection.
            weight (list of float): Per-atom weights, must be same length
                as the selection

        Returns:
            (list of float): RMSDs between residues in selections

        Raises:
            ValueError: if selections have differing numbers of atoms
            ValueError: if weight is the wrong size or contains non-floats
        """
        pass

    def fit(self, selection, weight=None):
        """
        Computes the transformation matrix required to produce a
        root-mean-square alignment of this selection to another.
        The format of the matrix is a 16-element tuple suitable for
        passing to the move() method. Weighting is optional.

        Args:
            selection (atomsel): Atom selection to fit to. Must have the
                same length as this selection.
            weight (list of float): Per-atom weights, must be the same
                length as this selection

        Returns:
            (tuple): Transformation matrix, in column major/fortran
                ordering

        Raises:
            ValueError: if weight is the wrong size or contains non-floats
            ValueError: if selection has a different number of atoms
        """
        pass

    def moveby(self, vector):
        """
        Moves the selection by the given vector.

        Args:
            vector (list): X,Y,Z translation vector

        Raises:
            ValueError: if no coordinates are loaded into this molecule
        """
        pass

    def move(self, matrix):
        """
        Applies a coordinate transformation to the selection.

        Args:
            matrix (16-tuple): Transformation matrix, of the form returned
                by fit(), in column major/fortran ordering

        Raises:
            ValueError: If matrix is the wrong size
            ValueError: If transformation could not be completed
        """
        pass

    def contacts(self, cutoff):
        """


