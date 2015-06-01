"""Druggability Index Analysis"""

__author__ = 'Ahmet Bakan'
__copyright__ = 'Copyright (C) 2010 Ahmet Bakan'

import numpy as np

import os
import gzip
import time
import os.path
import cPickle
import logging
import logging.handlers

from glob import glob


__all__ = ['DIA']

"""Druggability package related exceptions."""


class GridError(Exception):

    """Exception for errors related to handling grid files and data."""

    pass


class ProbeError(Exception):

    """Exception for errors related to handling probe grid and data."""

    pass


class DIAError(Exception):

    """Exception for errors related to druggability index calculations."""

    pass


"""This module defines a base class for other classes. Important features of
this base class are its logging and self pickling functionalities.

Classes:

* :class:`ABase`

Functions:

* :func:`get_logger`

"""

LOGGING_LEVELS = {'debug': logging.DEBUG,
                  'info': logging.INFO,
                  'warning': logging.WARNING,
                  'error': logging.ERROR,
                  'critical': logging.CRITICAL}
LOGGING_LEVELS.setdefault(logging.INFO)

SIGNATURE = '@>'

def _set_workdir(workdir):
    """Set a working directory, by creating if it doesn't exist."""
    if os.path.isabs(workdir):
        workdir = os.path.relpath(workdir)
    if not os.path.isdir(workdir):
        dirs = workdir.split(os.sep)
        for i in range(len(dirs)):
            dirname = os.sep.join(dirs[:i+1])
            try:
                if not os.path.isdir(dirname):
                    os.mkdir(dirname)
            except OSError:
                return os.getcwd()
    return os.path.join(os.getcwd(), workdir)

def get_logger(name, **kwargs):
    """Return a logger.

    :arg name: name of the logger instance
    :type name: str

    :keyword verbose: loglevel for console verbosity
    :type verbose: str, default is "info"

    :keyword writelog: control logging in a file
    :type writelog: bool, default is True

    :keyword workdir: location of logfile
    :type workdir: str, default is "."

    :keyword loglevel: loglevel for logfile verbosity
    :type loglevel: str, default is "debug"

    :keyword logfilemode: mode in which logfile will be opened
    :type logfilemode: str, default is "w"

    :keyword backupcount: number of old *name.log* files to save
    :type backupcount: int, default is 3

    ======== ==================================================================
    Loglevel Description
    ======== ==================================================================
    debug    Eveything will be printed on the colsole or written into logfile.
    info     Only brief information will be printed or written.
    warning  Only critical information will be printed or written.
    error    This loglevel is equivalent to *warning* in package.
    critical This loglevel is equivalent to *warning* in package.
    ======== ==================================================================


    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger


    console = logging.StreamHandler()
    console.setLevel(LOGGING_LEVELS[kwargs.get('verbose', 'info')])
    console.setFormatter(logging.Formatter(SIGNATURE + ' %(message)s'))
    logger.addHandler(console)

    if not (kwargs.has_key('writelog') and not kwargs['writelog']):
        logfilename = os.path.join(kwargs.get('workdir', '.'), name+'.log')
        rollover = False
        # if filemode='a' is provided, rollover is not performed
        if os.path.isfile(logfilename) and kwargs.get('filemode', None) != 'a':
            rollover = True
        logfile = logging.handlers.RotatingFileHandler(logfilename,
                    mode=kwargs.get('filemode', 'a'), maxBytes=0,
                    backupCount=kwargs.get('backupcount', 3))
        logfile.setLevel(LOGGING_LEVELS[kwargs.get('loglevel', 'debug')])
        logfile.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(logfile)
        if rollover:
            logger.debug('Saving this log file and starting a new one.')
            logfile.doRollover()
    return logger

class ABase(object):

    """A base class that provides logging and self pickling functionality.


    .. attribute:: name

       Name of the instance, which is frequently used as a prefix to output
       names.

    .. attribute:: workdir

       Working directory for the instance, into which outputs are written.

    .. attribute:: logger

       A Python logger for the instance. By default a log file is started for
       the instance with the name :file:`workdir/name.log`.

    """


    def __init__(self, name, **kwargs):
        """Instantiate class using an instance name.

        :arg name: name of the class instance
        :type name: str

        :keyword workdir: location of all outputs and logfile
        :type workdir: str, default is "."

        :keyword logger: logger instance to log actions and method calls
        :type logger: logging.Logger, default None

        Unless an existing logger is passed as a keyword argument, a new logger
        is started for the object. All keyword arguments are passed to the
        :meth:`get_logger` method.

        """
        self.name = name

        if kwargs.has_key('workdir'):
            workdir = kwargs['workdir']
        else:
            workdir = '.'

        kwargs['workdir'] = _set_workdir(workdir)

        if kwargs.has_key('logger'):
            self.logger = kwargs['logger']
        else:
            self.logger = get_logger(name, **kwargs)
        self.logger.info('{0:s} is initialized.'.format(self.name))
        self.set_workdir(workdir)

    def set_workdir(self, workdir):
        """Change working directory.

        If *workdir* does not exist, it is created.

        :arg workdir: new working directory
        :type workdir: str

        """
        self.workdir = _set_workdir(workdir)
        self.logger.info('{0:s} working directory is set to "{1:s}".'
                         .format(self.name, os.path.relpath(workdir)))

    def set_logger(self, **kwargs):
        """Setup a logger.

        This method can be used to reset current logger or to restart logger
        after the object is unpickled.

        All keyword arguments are passed to the :func:`get_logger` method.

        """
        if not kwargs.has_key('filemode'):
            kwargs['filemode'] = 'a'
        self.logger = get_logger(self.name, workdir=self.workdir, **kwargs)

    def pickle(self, filename=None, compress=True):
        """cPickle the object into a file with .dso(.gz) extension.

        Handler objects of logger attribute prevents pickling of this
        object. Hence, this method is defined. It temporarily removes
        handlers from the logger, pickles the object, and restores the
        handlers.

        To restore the object, use pickler method in Functions module.

        :arg filename: name of the file to dump object (without an extension)
        :type filename: str or None, default is :attr:`ABase.name`

        :arg compress: gzip output
        :type compress: bool, default is True

        """
        if filename is None:
            filename = os.path.join(self.workdir, self.name)
        filename += '.dso'
        if compress:
            filename += '.gz'
            out = gzip.open(filename, 'w')
        else:
            out = open(filename, 'w')

        # spare logger
        logger = self.logger
        self.logger = None

        cPickle.dump(self, out)
        out.close()

        # restore logger and kdtree
        self.logger = logger

        self.logger.info('{0:s} is cPickled into file {1:s}.'
                         .format(self.name, os.path.relpath(filename)))


"""Auxiliary functions for Druggability Index Analysis class.

Functions:

* :func:`pickler`
* :func:`get_histr`
* :func:`single_linkage`
* :func:`convert_dG_to_Kd`
* :func:`convert_Kd_to_dG`
* :func:`format_Kd`
* :func:`get_pdb_lines`

"""

GAS_CONSTANT = 0.0019858775                        # kcal/K/mol


def get_histr(array, bins=10, **kwargs):
    """Return histogram as a string for printing purposes.

    This function is making use of Numpy :func:`histogram` function.

    :arg array: data array
    :type array: array_like

    :arg bins: number of bins
    :type bins: int, default is 10

    :arg format: number format string
    :type format: str, default is {0:.2f}

    :arg orientation: horizontal ("h") or vertical ("v")
    :type orientation: str, default is "h"

    :arg line: line symbol
    :type line: str, defaults are "|" for vertical and "-" for horizontal

    :arg marker: line end symbol
    :type marker: str, default is "o"

    :arg label: axis label
    :type label: str

    :arg title: histogram title
    :type title: str

    """
    if isinstance(array, np.ndarray):
        if array.ndim > 1:
            raise ValueError('array must be 1-dimensional')
    counts, ranges = np.histogram(array, bins=bins)

    maxcount = max(counts)
    format = kwargs.get('format', '{0:.2f}')
    r_width = max([len(format.format(x)) for x in ranges])

    histr = ''
    marker = kwargs.get('marker', 'o')
    if kwargs.get('orientation', 'h')[0] == 'h':
        counts = counts[-1::-1]
        ranges = ranges[-1::-1]
        if kwargs.has_key('title'):
            histr += kwargs['title'].center(maxcount + r_width + 4) + '\n'
        if kwargs.has_key('label'):
            histr += kwargs['label'] + '\n'
        line = kwargs.get('line', '-')
        histr += ' ' * r_width + ' #' + '-' * maxcount + '-#\n'
        for i, count in enumerate(counts):
            histr += format.format(ranges[i]).strip().rjust(r_width) + ' |'
            histr += line * (count - 1) + marker * (count > 0)
            histr += ' ' * (maxcount - count + 1) + '|\n'

        histr += ' ' * r_width + ' #' + '-' * maxcount + '-#\n'
        histr += ' ' * r_width + ' 0'
        for i in range(0, maxcount+1, 5):
            if i > 0:
                histr += str(i).rjust(5)
    else:
        line = kwargs.get('line', '|')
        r_width += 1
        clen = len(str(maxcount))
        length = clen + 3 + r_width * bins + 2
        if kwargs.has_key('title'):
            histr += kwargs['title'].center(length) + '\n'

        histr += '#' * clen + ' --' + '-' * r_width * bins + '--' + '\n'

        for i in range(maxcount, 0, -1):
            histr += str(i).rjust(clen) + ' | '
            for j in range(bins):
                if counts[j] > i:
                    histr += line.center(r_width)
                elif counts[j] == i:
                    histr += marker.center(r_width)
                else:
                    histr += ' ' * r_width
            histr += ' |' + '\n'
        histr += ' ' * clen + ' --' + '-' * r_width * bins + '--' + '\n'
        histr += ' ' * clen + '   ' + ''.join(
                    [format.format(x).strip().center(r_width)
                        for x in ranges[:-1]]
                    )  + '\n'
        if kwargs.has_key('label'):
            histr += kwargs['label'].center(length) + '\n'

    return histr

def pickler(filename, obj=None, **kwargs):
    """cPickle/uncPickle an object to/from a gzipped file with the given name.

    If the unpickled object has "set_logger" method, it will be called.

    This function is defined to complement the
    :meth:`druggability.abase.ABase.pickle`
    method. It is recommended that ProDy objects are unpickled using this
    function.

    """
    if obj is None:
        if not os.path.isfile(filename):
            raise IOError('{0:s} is not found.'.format(filename))
        if filename.endswith('gz'):
            gzfile = gzip.open(filename)
        else:
            gzfile = open(filename)
        obj = cPickle.load(gzfile)
        gzfile.close()

        if isinstance(obj, ABase):
            fileloc = os.path.split(os.path.join(os.getcwd(), filename))[0]
            if obj.workdir != fileloc:
                obj.workdir = fileloc

            obj.set_logger(**kwargs)
            if obj.__dict__.has_key('name'):
                obj.logger.info('{0:s} has been reloaded.'.format(obj.name))
            else:
                print('{0:s} has been unpickled.'.format(filename))

            obj.set_workdir(obj.workdir)
        else:
            print('{0:s} has been unpickled.'.format(filename))
    else:
        try:
            dumps = cPickle.dumps(obj)
            gzfile = gzip.open(filename, 'w')
            gzfile.write(dumps)
            gzfile.close()
        except cPickle.UnpickleableError:
            raise TypeError('Object is unpickleable. If it has a '
                            '"pickle" method, try using it.')
        obj = None
    return obj

def single_linkage(dist, cutoff):
    """Return clusters determined by single-linkage agglomerative clustering.

    For speedup purposes, distance matrix may contain square-distances and
    cutoff distance may be cutoff-square.

    :arg dist: distance matrix, strictly lower triangular or a symmetric matrix
               with 0s along the diagonal are acceptable forms.
    :type dist: np.ndarray
    :arg cutoff: distance within which two cluster will be merged to make a new
                 cluster
    :type cutoff: float

    :return: an array of cluster assignments, i.e. items in the same cluster
             will have save cluster number
    :rtype: np.ndarray


    """
    if not isinstance(dist, np.ndarray):
        raise TypeError('dist must be of type numpy.ndarray')
    elif dist.ndim != 2:
        raise ValueError('dist must be a 2-dimensional array')
    elif dist.shape[0] != dist.shape[1]:
        raise ValueError('dist must be a square matrix')

    size = dist.shape[0]
    clust = [set([i]) for i in range(size)]
    for i in range(size-1):
        which = (dist[i+1:, i] <= cutoff).nonzero()[0] + i+1
        new = clust[i]
        for j in which:
            new = new.union(clust[j])
        for j in new:
            clust[j] = new
    clusters = - np.ones(size, 'int64')
    iclust = 0
    for i in range(size):
        if clusters[i] == -1:
            clusters[list(clust[i])] = iclust
            iclust += 1
    return clusters

def convert_dG_to_Kd(dG, temperature=300.0):
    """Return molar affinity for given binding free energy (kcal/mol)."""
    return np.exp(dG / GAS_CONSTANT / temperature)

def format_Kd(Kd):
    """Return formatted Kd."""
    if Kd < 1e-14:
        return Kd * 1e15, 'fM'
    elif Kd < 1e-11:
        return Kd * 1e12, 'pM'
    elif Kd < 1e-7:
        return Kd * 1e9, 'nM'
    elif Kd < 1e-4:
        return Kd * 1e6, 'uM'
    elif Kd < 1e-1:
        return Kd * 1e3, 'mM'
    else:
        return 0, '--'

def convert_Kd_to_dG(Kd, temperature=300.0):
    """Return binding free energy (kcal/mol) for given molar affinity."""
    return GAS_CONSTANT * temperature * np.log(Kd)

def get_pdb_lines(coordinates, **kwargs):
    """Return PDB lines for a given set of coordinates.

    :arg coordinates: Coordinate array, with shape (N,3). N is number of atoms.
    :type coordinates: numpy.ndarray

    :keyword atomnames: List of atom names.
    :type atomnames: array_like, default is ["CA", "CA", ...]

    :keyword resnames: List of residue names.
    :type resnames: array_like, default is ["GLY", "GLY", ...]

    :keyword chainids: List of chain ids.
    :type chainids: array_like, default is ["A", "A", ...]

    :keyword resids: List of residue numbers.
    :type resids: array_like, default is [1, 2, 3, ...]

    :keyword occupancy: Atomic occupancy values.
    :type occupancy: array_like, default is [1, 1, ...]

    :keyword bfactor: Atomic bfactor values.
    :type bfactor: array_like, default is [0, 0, ...]

    :keyword connect: PDB conect lines, as a list of tuples of atom indices.
    :type connect: array_like

    :keyword nonzero: Boolean option to skip atoms with zero occupancy
    :type nonzer: bool, default is True

    """
    if isinstance(coordinates, np.ndarray):
        if coordinates.ndim == 2:
            n_atoms = coordinates.shape[0]
        else:
            raise ValueError('coordinates must be 2-d. Given array is {0:d}-d.'
                            .format(coordinates.ndim))
    else:
        raise TypeError('coordinates must be of type numpy.ndarray.')

    atomnames = kwargs.get('atomnames', None)
    if atomnames is None:
        atomnames = ['CA'] * n_atoms
    elif not isinstance(atomnames, (np.ndarray, tuple, list)):
        raise TypeError('atomnames must be an array like object')
    elif len(atomnames) != n_atoms:
        raise ValueError('length of atomnames must match number of atoms')

    resnames = kwargs.get('resnames', None)
    if resnames is None:
        resnames = ['GLY'] * n_atoms
    elif not isinstance(resnames, (np.ndarray, tuple, list)):
        raise TypeError('resnames must be an array like object')
    elif len(resnames) != n_atoms:
        raise ValueError('length of resnames must match number of atoms')

    chainids = kwargs.get('chainids', None)
    if chainids is None:
        chainids = ['A'] * n_atoms
    elif not isinstance(chainids, (np.ndarray, tuple, list)):
        raise TypeError('chainids must be an array like object')
    elif len(chainids) != n_atoms:
        raise ValueError('length of chainids must match number of atoms')

    resids = kwargs.get('resids', None)
    if resids is None:
        resids = range(1, n_atoms+1)
    elif not isinstance(resids, (np.ndarray, tuple, list)):
        raise TypeError('resids must be an array like object')
    elif len(resids) != n_atoms:
        raise ValueError('length of resids must match number of atoms')

    occupancy = kwargs.get('occupancy', None)
    if occupancy is None:
        occupancy = np.ones(n_atoms)
    elif isinstance(occupancy, np.ndarray):
            bfactor = occupancy.flatten()
    elif not isinstance(occupancy, (tuple, list)):
        raise TypeError('occupancy must be an array like object')
    elif len(occupancy) != n_atoms:
        raise ValueError('length of occupancy must match number of atoms')

    bfactor = kwargs.get('bfactor', None)
    if bfactor is None:
        bfactor = np.zeros(n_atoms)
    elif isinstance(bfactor, np.ndarray):
            bfactor = bfactor.flatten()
    elif not isinstance(bfactor, (tuple, list)):
        raise TypeError('bfactor must be an array like object')
    elif len(bfactor) != n_atoms:
        raise ValueError('length of bfactor must match number of atoms')

    nonzero = kwargs.get('nonzero', False)
    if not isinstance(nonzero, bool):
        raise TypeError('nonzero must be of type bool')

    pdb = ''
    for i, xyz in enumerate(coordinates):
        if nonzero and not occupancy[i]:
            continue
        pdb += ('{0:6s}{1:5d}{2:2s}{3:3s}{4:1s}{5:4s}{6:1s}{7:4d}' +
           '{8:4s}{9:8.3f}{10:8.3f}{11:8.3f}{12:6.2f}{13:6.2f}' +
           '{14:6s}{15:4s}{16:2s}\n').format('ATOM  ', i+1, '  ',
            atomnames[i].ljust(3), ' ', resnames[i].ljust(4),
            chainids[i], int(resids[i]), '    ', float(xyz[0]), float(xyz[1]), float(xyz[2]),
            float(occupancy[i]), float(bfactor[i]), '      ', '    ',
            atomnames[i][0].rjust(2))

    conect = kwargs.get('connect', None)
    if conect is not None:
        if not isinstance(nonzero, list):
            raise TypeError('connect must be a list')
        for bond in conect:
            if not isinstance(bond, (tuple, list)):
                raise TypeError('items of connect must be array_like')
            elif len(bond) > 1:
                raise ValueError('items of connect must have length 2')
            pdb += 'CONECT{0:5d}{1:5d}\n'.format(bond[0], bond[1])
    return pdb


"""This module defines classes for parsing, smoothing and writing grid files.

In Druggability calculations, grid files contain probe atom counts (or
occupancy). This module defines a base class and file format specific
derived classes. These classes are direcly used by the
:mod:`druggability.probe` module.

Classes:

* :class:`Grid`
* :class:`OpenDX`
* :class:`XPLOR`

File Type Documentation
-----------------------

XPLOR Crystallographic map files
--------------------------------
This information is from XPLOR documentation:
    http://www.scripps.edu/rc/softwaredocs/msi/xplor981/formats.html

The X-PLOR program is able to write electron density map files in either a
binary format or in an ASCII format. The binary format is more compact and may
be read more quickly than the ASCII format but has the disadvantage that it may
not be readable when transferred between different kinds of computer. The ASCII
formatted file is written if the FORMatted keyword is set TRUE (the default);
setting FORMatted=FALSE will cause a binary map file to be written.

Map header

The X-PLOR map file begins with an eight-line header.
1.     Line 1
An empty line written by the `/ ` FORTRAN format descriptor in the formatted
map file.
2.     Lines 2- 5
Title information written as character strings. These lines are written as
80-character strings in the formatted file map.
3.     Line 6
A series of nine integers NA, AMIN, AMAX, NB, BMIN, BMAX, NC, CMIN, CMAX. The
values NA, NB and NC indicate the total number of grid points along the a,b,
and c cell edges. The items AMIN, AMAX, BMIN, BMAX, CMIN, CMAX indicate the
starting and stopping grid points along each cell edge in the portion of the
map that is written. In the formatted map file this line is written using the
FORTRAN format statement (9I8).
4.     Line 7
A series of six double-precision items corresponding to the crystal cell
dimensions a, b, c, alpha, beta, gamma. In the formatted map file these items
are written using the FORTRAN format statement (6E12.5).
5.     Line 8
A three-letter character string which always reads `ZXY'.

Density array
Following the map header, the density matrix is then written section-by-section
with c moving slowest (in z-sections). Each section of the density map is
preceded by a section number.
Thus, for the formatted map file each section of the density map is written
using FORTRAN statements of the type

WRITE(OUNIT,'(I8)') KSECT

WRITE(OUNIT,'(6E12.5)') ((SECTON(I,J),I=1,ISECT),J=1,JSECT)

and the resulting map is written with six pixels on each line. The binary
format is identical except the format statements are missing, so that each
line that is written contains the entire length of map along the `fast'
(a-axis) direction.


Map footer

Two lines follow the density array.

1.     Line 1
The integer `-9999' is always written. For the formatted map file, The FORTRAN
format statement (I8) is used to write this value.

2.     Line 2
Two double-precision items corresponding to the average electron density and
the standard deviation of the map. For the formatted map file these items are
written using the FORTRAN format statement (2(E12.4,1X)).


OpenDX scalar data
------------------
This information is from APBS website:
    <http://www.poissonboltzmann.org/file-formats/mesh-and-data-formats/
    opendx-scalar-data>

We output most discretized scalar data (e.g., potential, accessibility,
etc.) from APBS in the data format used by the OpenDX software package.
The OpenDX data format is very flexible; the following sections describe
the application of this format for APBS multigrid and finite element
datasets.
The multigrid data format has the following form::

   # Comments
   object 1 class gridpositions counts nx ny nz
   origin xmin ymin zmin
   delta hx 0.0 0.0
   delta 0.0 hy 0.0
   delta 0.0 0.0 hz
   object 2 class gridconnections counts nx ny nz
   object 3 class array type double rank 0 items n data follows
   u(0,0,0) u(0,0,1) u(0,0,2)
   ...
   u(0,0,nz-3) u(0,0,nz-2) u(0,0,nz-1)
   u(0,1,0) u(0,1,1) u(0,1,2)
   ...
   u(0,1,nz-3) u(0,1,nz-2) u(0,1,nz-1)
   ...
   u(0,ny-1,nz-3) u(0,ny-1,nz-2) u(0,ny-1,nz-1)
   u(1,0,0) u(1,0,1) u(1,0,2)
   ...
   attribute "dep" string "positions"
   object "regular positions regular connections" class field
   component "positions" value 1
   component "connections" value 2
   component "data" value 3

The variables in this format have been shown in bold and include
Comments
Any number of comment lines, each line starting with the "#" symbol
nx ny nz
The number of grid points in the x-, y-, and z-directions.
xmin ymin zmin
The coordinates of the grid lower corner.
hx hy hz
The grid spacings in the x-, y-, and z-directions.
n
The total number of grid points; n = nx * ny * nz
u(*,*,*)
The data values, ordered with the z-index increasing most quickly,
followed by the y-index, and then the x-index.

"""

class Grid(object):

    """Base class for manipulating grid data.

    An important grid attribute is :attr:`state`. After the grid file is
    parsed, this is set to *original*. This attribute is compared or changed
    when two grids are added or a grid :meth:`smooth` is called.

    .. attribute:: filename

       Name of the grid file.

    .. attribute:: format

       Grid file format.

    .. attribute:: name

       Name of the instance, which is deduced from the filename. Name is used
       as a prefix to output file names.

    .. attribute:: array

       3-dimentional Numpy array that holds grid data.

    .. attribute:: offset

       Offset of the origin of the grid from origin of the Cartesian coordinate
       space.

    .. attribute:: shape

       Shape of the data array.

    .. attribute:: spacing

       Resolution of the grid.

    .. attribute:: state

       State of the grid. *original* and *smooth* are predefined states.

    """


    def __init__(self, filename=None):
        """Instantiate grid from a file.

        A grid maybe initialized without a filename. If a filename is provided,
        file format specific derivatives of this class parse the file at
        initialization. To delay this, one can initialize a grid without a
        filename and then use :meth:`parse` method to parse the grid file.

        """

        self.name = None
        self.filename = None
        self.array = None
        self.state = None
        self.offset = None
        self.spacing = None
        self.shape = None

    def __repr__(self):
        return 'Grid {0:s} (file: {1:s}) in {2:s} state'.format(self.name,
                                                    self.filename, self.state)

    def __add__(self, other):
        """Compare grid attributes to determine if grids can be added.

        File type specific Grid derivatives need to implement method for
        adding grid data array.

        Before adding data arrays of the grids, four attributes are compared.
        These are :attr:`state`, :attr:`shape`, :attr:`spacing`, and
        :attr:`offset`. All of these attributes must have the same values.

        """
        if self.state != other.state:
            raise GridError('both grids must be at the same state')
        elif self.shape != other.shape:
            raise GridError('both grids must have the same shape')
        elif all(self.spacing != other.spacing):
            raise GridError('both grids must have the same spacing')
        elif all(self.offset != other.offset):
            raise GridError('both grids must have the same offset')
        else:
            return True


    def _smooth(self):
        """Smooth grid array by averaging over neighboring grid elements."""
        self.array = (self.array[0:-2, 0:-2, 0:-2] +
                      self.array[0:-2, 0:-2, 1:-1] +
                      self.array[0:-2, 0:-2, 2:  ] +
                      self.array[0:-2, 1:-1, 0:-2] +
                      self.array[0:-2, 1:-1, 1:-1] +
                      self.array[0:-2, 1:-1, 2:  ] +
                      self.array[0:-2, 2:,   0:-2] +
                      self.array[0:-2, 2:,   1:-1] +
                      self.array[0:-2, 2:,   2:  ] +
                      self.array[1:-1, 0:-2, 0:-2] +
                      self.array[1:-1, 0:-2, 1:-1] +
                      self.array[1:-1, 0:-2, 2:  ] +
                      self.array[1:-1, 1:-1, 0:-2] +
                      self.array[1:-1, 1:-1, 1:-1] +
                      self.array[1:-1, 1:-1, 2:  ] +
                      self.array[1:-1, 2:,   0:-2] +
                      self.array[1:-1, 2:,   1:-1] +
                      self.array[1:-1, 2:,   2:  ] +
                      self.array[2:,   0:-2, 0:-2] +
                      self.array[2:,   0:-2, 1:-1] +
                      self.array[2:,   0:-2, 2:  ] +
                      self.array[2:,   1:-1, 0:-2] +
                      self.array[2:,   1:-1, 1:-1] +
                      self.array[2:,   1:-1, 2:  ] +
                      self.array[2:,   2:,   0:-2] +
                      self.array[2:,   2:,   1:-1] +
                      self.array[2:,   2:,   2:  ]) / 27.0
        self.state = 'smooth'

    def parse(self, filename):
        """File type specific method for parsing grid data."""
        pass

    def write(self, filename=None):
        """File type specific method for writing grid data."""
        pass

    def smooth(self):
        """File type specific method for averaging grid data.

        Smoothing is performed by assigning a grid element the value found by
        averaging values of itself and its neighboring grid elements. After
        this operation grid becomes smaller by two elements along each
        direction.

        Calling this method changes the :attr:`state` attribute of the grid
        from *original* to *smooth*.

        """
        pass


class XPLOR(Grid):

    """A class to manipulate XPLOR formatted contour file.

    This class is tested using grid files outputed by ptraj from `AmberTools
    <http://ambermd.org/>`_.

    """


    def __init__(self, filename=None):
        """Instantiation arguments are passed to :class:`Grid`"""
        Grid.__init__(self, filename)
        self.format = 'Xplor'
        self._size = None
        self._ignored_lines = None
        self._first = None
        self._last = None
        self._angles = None

        if filename:
            self.parse(filename)


    def __add__(self, other):
        if not Grid.__add__(self, other):
            raise GridError('{0:s} and {1:s} cannot be added'
                            .format(self, other))
        grid = XPLOR(None)
        grid.name = '(' + self.name + ' + '+ other.name + ')'
        grid.array = self.array + other.array
        grid.state = self.state

        grid.offset = self.offset
        grid.spacing = self.spacing
        grid.shape = self.shape

        grid._ignored_lines = self._ignored_lines
        grid._size = self._size
        grid._first = self._first
        grid._last = self._last
        grid._angles = self._angles
        return grid

    def smooth(self):
        """Smooth grid and change grid attributes and state."""
        self._smooth()
        self.shape = np.shape(self.array)
        self.offset += self.spacing

        self._size -= self.spacing * 2
        self._first += 1
        self._last -= 1


    def parse(self, filename):
        """Parse grid data from file."""
        if not os.path.isfile(filename):
            raise IOError('{0:s} not found'.format(filename))
        self.filename = filename
        self.name = os.path.splitext(os.path.split(filename)[1])[0]

        xplor_file = open(filename)
        xplor = xplor_file.read()
        xplor_file.close()

        xyz = xplor.index('ZYX')
        lines = xplor[:xyz].split('\n')

        self._ignored_lines = lines[:3] #Hold lines that're not used
        line = lines[3].split() #Parse indexing related numbers
        self.shape = np.array((int(line[0]), int(line[3]), int(line[6])),
                              'int64')
        self._first = np.array((int(line[1]), int(line[4]), int(line[7])),
                              'int64')
        self._last = np.array((int(line[2]), int(line[5]), int(line[8])),
                              'int64')

        line = lines[4].split()
        self._size = np.array((float(line[0]), float(line[1]), float(line[2])),
                             'd')
        self.spacing = self._size / self.shape
        self._angles = (float(line[3]), float(line[4]), float(line[5]))

        self.offset = (self._first - 0.5) * self.spacing

        array = np.fromstring(xplor[xyz+4:], dtype='d', sep=' ')
        self.array = np.zeros(self.shape, 'd')
        yxshape = (self.shape[1], self.shape[0])
        length = yxshape[0] * yxshape[1] + 1
        for k in range(self.shape[2]):
            self.array[:, :, k] = array[length*k+1:length*(k+1)].reshape(
                                                                    yxshape).T
        self.state = 'original'

    def write(self, filename=None):
        """Write grid data into a file.

        If a filename is not provided, gridname_state.xplor will be used.

        """
        if filename is None:
            filename = os.path.splitext(self.filename)
            filename = filename[0] + '_' + self.state + filename[1]

        xplor_file = open(filename, 'w')
        for line in self._ignored_lines:
            xplor_file.write(line+'\n')
        xplor_file.write(('{0[0]:8d}{1[0]:8d}{2[0]:8d}'
                          '{0[1]:8d}{1[1]:8d}{2[1]:8d}'
                          '{0[2]:8d}{1[2]:8d}{2[2]:8d}\n').format(
                                          self.shape, self._first, self._last))
        xplor_file.write(('{0[0]:12.3f}{0[1]:12.3f}{0[2]:12.3f}'
                          '{1[0]:12.3f}{1[1]:12.3f}{1[2]:12.3f}\n').format(
                                                     self._size, self._angles))
        xplor_file.write('ZYX\n')

        format_ = ''
        for i in range(self.shape[0]):
            if i != 0 and i % 6 == 0:
                format_ += '\n'
            format_ += '{0['+str(i)+']:12.5f}'
        else:
            if i % 6 != 0:
                format_ += '\n'

        for k in range(self.shape[2]):
            xplor_file.write('{0:8d}\n'.format(k + self._first[2]))
            for j in range(self.shape[1]):
                xplor_file.write(format_.format(self.array[:, j, k]))
        xplor_file.close()
        return filename

class OpenDX(Grid):

    """A class to manipulate OpenDX scalar data files.

    This classes is tested using grid files outputed by volmap in `VMD
    <http://www.ks.uiuc.edu/Research/vmd/current/ug/>`_.

    Additional relevant information on this file format may be found here
    `APBS website <http://www.poissonboltzmann.org/file-formats/
    mesh-and-data-formats/opendx-scalar-data>`_

    """


    def __init__(self, filename=None):
        """Instantiation arguments are passed to :class:`Grid`"""
        Grid.__init__(self, filename)
        self.format = 'OpenDX'
        self._origin = None
        self._comments = None

        if filename:
            self.parse(filename)

    def __add__(self, other):
        if not Grid.__add__(self, other):
            raise GridError('{0:s} and {1:s} cannot be added'
                            .format(self, other))
        grid = OpenDX(None)
        grid.name = '(' + self.name + ' + '+ other.name + ')'
        grid.array = self.array + other.array
        grid.state = self.state

        grid.offset = self.offset
        grid.spacing = self.spacing
        grid.shape = self.shape

        grid._comments = self._comments
        grid._origin = self._origin
        return grid


    def parse(self, filename):
        """Parse grid data from file."""

        if not os.path.isfile(filename):
            raise IOError('{0:s} not found'.format(filename))
        self.filename = filename
        self.name = os.path.splitext(os.path.split(filename)[1])[0]

        opendx_file = open(filename)
        opendx = opendx_file.read()
        opendx_file.close()

        lindex = opendx.index('data follows') + len('data follows') + 1
        lines = opendx[:lindex].split('\n')
        self._comments = []
        self.spacing = np.zeros(3, 'd')
        for line in lines:
            if line.startswith('#'):
                self._comments.append(line)
            elif line.startswith('object 1'):
                items = line.split()
                self.shape = (int(items[-3]), int(items[-2]), int(items[-1]))
            elif line.startswith('origin'):
                items = line.split()
                self._origin = np.array(items[1:], 'd')
            elif line.startswith('delta'):
                items = line.split()
                self.spacing += np.array(items[1:], 'd')
        rindex = opendx.rindex('object')
        self._comments.append(opendx[rindex:].strip())
        self.offset = self._origin - self.spacing / 2
        self.array = np.fromstring(opendx[lindex:rindex], dtype='d', sep=' '
                       ).reshape((self.shape[0], self.shape[1], self.shape[2]))

        self.state = 'original'


    def smooth(self):
        """Smooth grid and change grid attributes and state."""
        self._smooth()
        self.shape = self.array.shape
        self.offset += self.spacing

        self._origin += self.spacing

    def write(self, filename=None):
        """Write grid data into a file.

        If a filename is not provided, gridname_state.dx will be used.

        """
        if filename is None:
            filename = os.path.splitext(self.filename)
            filename = filename[0] + '_' + self.state + filename[1]

        opendx = open(filename, 'w')
        opendx.write('{0:s} modified by Druggability\n'
                     .format(self._comments[0]))
        opendx.write('object 1 class gridpositions counts {0[0]:d} {0[1]:d} '
                     '{0[2]:d}\n'.format(self.shape))
        opendx.write('origin {0[0]:.9g} {0[1]:.9g} {0[2]:.9g}\n'
                     .format(self._origin))
        opendx.write('delta {0:.9g} 0 0\n'.format(self.spacing[0]))
        opendx.write('delta 0 {0:.9g} 0\n'.format(self.spacing[1]))
        opendx.write('delta 0 0 {0:.9g}\n'.format(self.spacing[2]))
        opendx.write('object 2 class gridconnections counts {0[0]:d} {0[1]:d} '
                     '{0[2]:d}\n'.format(self.shape))
        length = self.shape[0]*self.shape[1]*self.shape[2]
        opendx.write('object 3 class array type double rank 0 items {0:d} data'
                     ' follows\n'.format(length))

        array = self.array.flatten()
        string = ''
        times = length / 9
        for i in range(times):
            string += ('{0[0]:.9g} {0[1]:.9g} {0[2]:.9g}\n'
                       '{0[3]:.9g} {0[4]:.9g} {0[5]:.9g}\n'
                       '{0[6]:.9g} {0[7]:.9g} {0[8]:.9g}\n'
                      ).format(
                       array[i*9:i*9+9])
        length = length - times * 9
        times = length / 3
        for i in range(times):
            string += '{0[0]:.9g} {0[1]:.9g} {0[2]:.9g}\n'.format(
                        array[i*3:i*3+3])
        length = length - times * 3
        if length == 2:
            string += '{0[0]:.9g} {0[1]:.9g}\n'.format(array[-2:])
        elif length == 1:
            string += '{0:.9g}\n'.format(array[-1])
        opendx.write(string)
        opendx.write('\n{0:s}\n'.format(self._comments[1]))
        opendx.close()
        return filename


"""This module defines a class to analyze probe grid file.

This class is used by :class:`druggability.dia.DruggabilityIndexAnalysis`,
which will frequently be referred to as DIA.

Classes:

* :class:`Probe`

Functions:

* :func:`get_expected_occupancy`

For each probe type, a probe card is defined as a dictionary. Probe cards
contain information on the chemical identity and physical properties of the
probe molecules:

* name: full chemical name
* radius: average distance of the central atom other (moleule) heavy atoms
* atomname: name of the central atom, used when writing PDB files
* n_atoms: number of heavy atoms
* charge: charge of the probe

For example, isopropanol card is defined as follows::

    IPRO = {'name': 'isopropanol',
        'radius': 3.99,
        'atomname': 'C2',
        'n_atoms': 4,
        'charge': 0}

And then this card is registered as::

    PROBE_CARDS['IPRO'] = IPRO

"""

# Probe Cards are defined
IPRO = {'name': 'isopropanol',
        'radius': 2.564, #V=70.602 3.24,
        'atomname': 'C2',
        'n_atoms': 4,
        'charge': 0}
IBUT = {'name': 'isobutane',
        'radius': 2.664, #V=79.146 3.40,
        'atomname': 'C2',
        'n_atoms': 4,
        'charge': 0}
IPAM = {'name': 'isopropylamine',
        'radius': 2.603, #V=73.873 3.18,
        'atomname': 'C2',
        'n_atoms': 4,
        'charge': +1}
ACET = {'name': 'acetate',
        'radius': 2.376, #V=56.197 2.94,
        'atomname': 'C2',
        'n_atoms': 4,
        'charge': -1}
ACAM = {'name': 'acetamide',
        'radius': 2.421, #V=59.468 3.14,
        'atomname': 'C2',
        'n_atoms': 4,
        'charge': 0}
ALL = {'name': 'all-probes',
        'radius': 2.564*.6 + 2.664*.1 + 2.603*.1 + 2.376*.1 + 2.421*.1,
        'atomname': 'C2',
        'n_atoms': None,
        'charge': 0}

PROBE_CARDS = {'IPRO': IPRO, 'IBUT': IBUT, 'IPAM': IPAM, 'ACET': ACET,
               'ACAM': ACAM, 'ALL': ALL,
               'ACTT': ACET,
               'PRO2': IPRO} # included for backwards compatibility

def get_expected_occupancy(grid_spacing):
    """Return number of expected probes in a rectangular grid element.

    Calculation is based on a reference simulation of water and probe mixture
    (see :ref:`mixture`).

    """
    gs3 = grid_spacing * grid_spacing * grid_spacing
    n_probes = 343                # of isopropanols
    #nW = 6860                    # of waters
    ave_vol = 240934.57           # from a 10 ns reference simulation
    #return (nW*gs3/ave_vol, nI*gs3/ave_vol)
    return n_probes * gs3 / ave_vol


class Probe(object):

    """A class to manipulate grid based probe count data.

    Probe count data typically comes from molecular dynamic simulations, and
    may be present in different data file formats. This class recognizes grid
    file formats for which a grid class is defined in the
    :mod:`druggability.grid` module.

    .. attribute:: dia

       DIA instance that the probe grid data belongs to.

    .. attribute:: type

       Type of the probe molecule, which is the residue name used in the force
       field definition of the probe.

    .. attribute:: grid

       A :class:`druggability.grid.Grid` instance that holds probe grid data.

    .. attribute:: name

       Full chemical name of the probe molecule.

    .. attribute:: radius

       An effective radius for the probe molecule.

    .. attribute:: atomname

       Name of the atom that represents the probe. This is usually the name
       of the central carbon atom.

    .. attribute:: charge

       Charge of the probe molecule.

    """


    def __init__(self, dia, probe_type, grid):
        """Instantiate with a DIA instance, probe type and grid.

        At instantiation, passing a DIA instance is required. Many operations
        over a probe object depends on the attributes of the DIA instance.

        :arg dia: DIA instance
        :type dia: :class:`druggability.dia.DruggabilityIndexAnalysis`

        :arg probe_type: probe type, see the following table
        :type probe_type: str

        :arg grid: grid filename or Grid instance
        :type grid: str or :class:`druggability.grid.Grid`

        ========== =============
        Probe Type Chemical name
        ========== =============
        IPRO, PRO2 isopropanol
        IBUT       isobutane
        IPAM       isopropylamine
        ACET       acetate
        ACAM       acetamide
        ========== =============

        Grid file types handled based on their extensions are:

        * .xplor (ptraj output grid file)
        * .dx (VMD output grid file)

        """
        if not probe_type in PROBE_CARDS:
            raise ProbeError('{0:s} is not a known probe type.'
                                 .format(probe_type))
        self.dia = dia
        self.type = probe_type
        card = PROBE_CARDS[self.type]
        self.name = card['name']
        self.radius = card['radius']
        self.atomname = card['atomname']
        self.charge = card['charge']

        if isinstance(grid, str):
            start = time.time()
            if grid.endswith('.xplor'):
                self.dia.logger.info('Parsing Xplor file {0:s}.'
                                     .format(os.path.relpath(grid)))
                self.grid = XPLOR(grid)
            elif grid.endswith('.dx'):
                self.dia.logger.info('Parsing OpenDX file {0:s}.'
                                     .format(os.path.relpath(grid)))
                self.grid = OpenDX(grid)
            else:
                raise ProbeError('Grid file {0:s} is not in a recognized '
                    'format (.xplor or .dx)'.format(grid))
            self.dia.logger.info('{0:s} was parsed in {1:.2f}s'
                                 .format(self.grid.name, time.time()-start))
        else:
            self.grid = grid

        self.expected_population = get_expected_occupancy(self.grid.spacing[0]
                                    ) #* self.fraction

    def __repr__(self):
        return 'Probe {0:s} ({1:s}) (grid: {2:s})'.format(self.name, self.type,
                                                          self.grid.filename)

    def write_grid(self, what=None, filename=None, smooth=True):
        """Write original, smoothed, free-energy, or enrichment grid.

        Output file content is determined by *what* argument. If *None*, is
        provided grid original data is outputed.

        By default, grid data is smoothed by calling the
        :meth:`druggability.grid.Grid.smooth` method of grid instance.

        :arg what: original (None), enrichment, free-energy
        :type what: str or None

        :arg filename: output filename
        :type filename: str

        :arg smooth: smooth grid data before writing grid file
        :type smooth: bool, default is True

        """
        if what is not None and not what in ['enrichment', 'free-energy']:
            raise ProbeError('{0:s} is not recognized as a valid grid type'
                             .format(what))
        if smooth and self.grid.state == 'original':
            self.grid.smooth()
        if what is None:
            what = self.grid.state

        start = time.time()
        self.dia.logger.info('Writing {0:s} {1:s} grid in {2:s}.'
                         .format(self.grid.name, what, self.grid.format))

        if what in ['origional', 'smooth']:
            filename = self.grid.write(filename)
        else:
            array = self.grid.array
            state = self.grid.state
            self.grid.array = self.grid.array / \
                              self.dia.parameters['n_frames'] / \
                              self.expected_population
            if what == 'enrichment':
                self.grid.state = 'enrichment'
                filename = self.grid.write(filename)
            else:
                self.grid.state = 'free-energy'
                which = self.grid.array <= 1
                self.grid.array[which] = 0
                which = self.grid.array > 1
                self.grid.array[which] = -convert_Kd_to_dG(
                                            self.grid.array[which])
                filename = self.grid.write(filename)
            self.grid.array = array
            self.grid.state = state
        self.dia.logger.info('{0:s} was written in {1:.2f}s'
                         .format(os.path.relpath(filename), time.time()-start))


"""Class to analyze druggability simulation trajectories.

Classes:

* :class:`DruggabilityIndexAnalysis`

Algorithmic details and theoretical basis of DIA is explained in
:ref:`methodology`.


DIA Parameters
-------------------------------------------------------------------------------

A table of parameters required for druggability index analysis is provided
below. User may set these parameters in a number of ways, including passing
them in a file, or passing them using a method in an interactive Python
session. See the documentation on :class:`DruggabilityIndexAnalysis`
for available options to set these parameters.

============ ======= ======== =================================================
Parameter    Default Unit     Description
============ ======= ======== =================================================
n_frames                      number of simulation frames used when
                              generating probe grid data

                              When VMD volmap plug-in is used, this parameter
                              needs to be set to 1, since volmap
                              calculates a frame averaged value. When
                              AmberTools ptraj module is used, the
                              actual number of frames is required,
                              since ptraj does not average over frames.

temperature          K        simulation temperature

delta_g      -1.239  kcal/mol binding free energy limit for hotspots

                              Hotspots with binding free energy below the value
                              set by :attr:`delta_g` will be used for assessing
                              druggability. A way to think of this parameter
                              may be in terms of ligand efficiency.
                              Currently available probes have 4 heavy atoms:
                              :math:`-delta_g / 4 = 0.31 LE`.

n_probes     6                number of probes to merge to make a drug-size
                              molecule mimic

                              Merging 6 hotspots puts together 24 heavy atoms.
                              Drug-like molecules have have 32-40 heavy atoms.
                              The difference is assumed to be atoms serving as
                              scaffolds without contributing to affinity.

min_n_probes 5                minimum number of probes in an acceptable
                              solution

merge_radius 6.0     A        hotspot merge distance

                              This radius is used when clustering hotspots
                              and merging them to make up a drug-size molecule.

low_affinity 10      uM       lower binding affinity limit

                              Potential binding sites with predicted affinity
                              better than :attr:`low_affinity` will be
                              reported.

n_solutions  3                number of drug-size solutions to report for
                              each potential binding site

                              This parameter may be set indefinetely large
                              in order to see all solutions in a potential
                              binding site.

max_charge   2       e        maximum absolute charge on the drug-size molecule

n_charged    3                number of charged hotspots in a solutions
============ ======= ======== =================================================

Note that some of the parameters do not have default values. User needs
to set at least their (:attr:`n_frames` and :attr:`temperature`) values
using :meth:`DruggabilityIndexAnalysis.set_parameters` method.
Where a unit is specified, user provided parameteer value will be assumed
to be in the specified unit.

"""


def _grow_molecules(alist, seed, dist, cutoff, size, dG):
    """Grow molecules starting from a seed hotspot.

    :arg alist: list of grown molecules
    :arg seed: list of indices
    :arg dist: distance matrix
    :arg cutoff: merge distance
    :arg size: maximum size

    """
    #mkp print alist, seed
    #mkp raw_input()
    if len(seed) >= size:
        alist.append(seed)
        return alist
    else:
        which = np.zeros(dist.shape[0])
        for i in seed:
            which += dist[i, :] <= cutoff
        which[seed] = False
        which = which.nonzero()[0]
        if len(which) == 0:
            return alist
        #i = which[(dG[which] == np.min(dG[which])).nonzero()[0]][0]
        #_grow_molecules(alist, seed + [i], dist, cutoff, size, dG)
        for i in which[:3]:
            _grow_molecules(alist, seed + [i], dist, cutoff, size, dG)
    return alist

class DIA(ABase):

    """Analyze druggability of proteins based on MD simulation data.

    .. attribute:: parameters

       A dictionary that keeps parameter values.

    .. attribute:: hotspots

       A Numpy array instance that keeps hotspot data. Each row corresponds to
       a hotspot. Columns correspond to:

       0. binding free energy (kcal/mol)
       1. X coordinate (A)
       2. Y coordinate (A)
       3. Z coordinate (A)
       4. probe identifier
       5. fractional occupancy
       6. charge (averaged by occupancy)
       7. effective radius
       8. cluster identifier

       First 8 columns are obtained by calling :meth:`identify_hotspots`
       method. Last column is obtained by calling :meth:`assess_druggability`
       method.


    """


    def __init__(self, name, **kwargs):
        """Initialize with a name and keyword arguments.

        All keyword arguments are passed to :class:`druggability.abase.ABase`.

        """
        ABase.__init__(self, name, **kwargs)

        self._probes = []
        self._dict = {}
        self._all = None
        self.logger.info('Druggability Analysis {0:s} is initialized.'
                         .format(name))

        self.parameters = {}

        self.parameters['temperature'] = None
        self.parameters['n_frames'] = None

        self.parameters['delta_g'] = -1.0

        self.parameters['n_probes'] = 6
        self.parameters['min_probes'] = 5
        self.parameters['merge_radius'] = 6.5
        self.parameters['n_solutions'] = 3
        self.parameters['low_affinity'] = -6.8589778920184257
        self.parameters['max_charge'] = 2.0
        self.parameters['n_charged'] = 3


        self.hotspots = None

    def __repr__(self):
        return 'Druggability analysis {0:s} containing {1:d} probes.'.format(
                self.name, len(self._probes))

    def set_parameters(self, **kwargs):
        """Set druggability analysis parameters.

        :mod:`druggability.dia` module documentation describes parameters
        and their default values.

        """
        if kwargs.has_key('temperature'):
            self.parameters['temperature'] = float(kwargs['temperature'])
            kwargs.pop('temperature')
            self.logger.info('Parameter: temperature {0:.2f} K'.format(
                                            self.parameters['temperature']))

        if kwargs.has_key('n_frames'):
            self.parameters['n_frames'] = int(kwargs['n_frames'])
            kwargs.pop('n_frames')
            self.logger.info('Parameter: n_frames {0:d}'.format(
                                                self.parameters['n_frames']))

        if kwargs.has_key('delta_g'):
            self.parameters['delta_g'] = float(kwargs['delta_g'])
            kwargs.pop('delta_g')
            self.logger.info('Parameter: delta_g {0:.3f} kcal/mol'
                             .format(self.parameters['delta_g']))

        if kwargs.has_key('n_probes'):
            self.parameters['n_probes'] = int(kwargs['n_probes'])
            kwargs.pop('n_probes')
            self.logger.info('Parameter: n_probes {0:d}'.format(
                                                self.parameters['n_probes']))

        if kwargs.has_key('min_n_probes'):
            self.parameters['min_n_probes'] = int(kwargs['min_n_probes'])
            kwargs.pop('min_n_probes')
            self.logger.info('Parameter: min_n_probes {0:d}'.format(
                                            self.parameters['min_n_probes']))
        if kwargs.has_key('n_solutions'):
            self.parameters['n_solutions'] = int(kwargs['n_solutions'])
            kwargs.pop('n_solutions')
            self.logger.info('Parameter: n_solutions {0:d}'.format(
                                            self.parameters['n_solutions']))


        if kwargs.has_key('merge_radius'):
            self.parameters['merge_radius'] = float(kwargs['merge_radius'])
            kwargs.pop('merge_radius')
            self.logger.info('Parameter: merge_radius {0:.1f} A'.format(
                                            self.parameters['merge_radius']))

        if kwargs.has_key('low_affinity'):
            self.parameters['low_affinity'] = convert_Kd_to_dG(
                                float(kwargs['low_affinity']) * 1e-6)
            kwargs.pop('low_affinity')
            self.logger.info('Parameter: low_affinity {0[0]:.2f} '
            '{0[1]:s}'.format(format_Kd(convert_dG_to_Kd(
                                            self.parameters['low_affinity']))))
        if kwargs.has_key('max_charge'):
            self.parameters['max_charge'] = float(kwargs['max_charge'])
            kwargs.pop('max_charge')
            self.logger.info('Parameter: max_charge {0:.1f} e'.format(
                                                self.parameters['max_charge']))

        if kwargs.has_key('n_charged'):
            self.parameters['n_charged'] = int(kwargs['n_charged'])
            kwargs.pop('n_charged')
            self.logger.info('Parameter: n_charged {0:d}'.format(
                                                self.parameters['n_charged']))


        while kwargs:
            key, value = kwargs.popitem()
            self.logger.warning('Parameter: {0:s} ({1:s}) is not a valid '
                                'parameter.'.format(key, str(value)))


    def parse_parameter_file(self, filename):
        """Parse parameters and probe grid types and filenames from a file.

        Example file content::

            n_frames 8000
            # Some comment
            temperature 300 K
            probe IPRO grid_IPRO.xplor 0.5
            probe ACAM grid_ACAM.xplor 0.5
            delta_g -1.0 kcal/mol
            n_probes 6
            # Another comment

        Note that all possible parameters are not included in this example.
        See documentation for "set_parameters" method.

        """
        kwargs = {}
        inp = open(filename)
        probes = []
        for line in inp:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            items = line.split()
            if items[0] == 'probe':
                probes.append((items[1], items[2]))
            else:
                if kwargs.has_key(items[0]):
                    raise DIAError('{0:s} parameter is repeated in {1:s}'
                                   .format(items[0], filename))
                kwargs[items[0]] = items[1]
        self.logger.info('Setting parameters from file {0:s}.'
                         .format(filename))
        self.set_parameters(**kwargs)
        self.logger.info('{0:d} probe files will be parsed.'
                         .format(len(probes)))
        for probe in probes:
            self.add_probe(probe[0], probe[1])


    def add_probe(self, probe_type, filename):
        """Add a probe grid file to the analysis.

        :arg probe_type: probe type is the residue name of the probe used by
                         the force field
        :type probe_type: str
        :arg filename: Path to the file that contains grid data for the probe.
                       Acceptable file formats are XPLOR and OpenDX.
        :type filename: str

        """
        if not PROBE_CARDS.has_key(probe_type):
            raise DIAError('Probe type {0:s} is not recognized.'
                           .format(str(probe_type)))
        for probe in self._probes:
            if probe.grid.filename == filename:
                raise DIAError('Probe grid file {0:s} has already been added '
                               'to the analysis.'.format(filename))
        if self._dict.has_key(probe_type):
            raise DIAError('Probe type {0:s} has already been added '
                           'to the analysis.'.format(probe_type))


        self._probes.append(Probe(self, probe_type, filename))
        self._dict[probe_type] = self._probes[-1]
        self._dict[self._probes[-1].name] = self._probes[-1]


    def get_probe(self, probe=None):
        """Return probe instance for a given probe name or type.

        Probe instance for given name (e.g. *isopropanol*) or probe type (e.g.
        *IPRO*) will be return. If *None* is provided, probe instance for
        combined probe data will be returned if it is available.

        """

        if probe is None:
            if self._all is not None:
                return self._all
            else:
                DIAError('Combined probe data is not available yet.')
        else:
            if self._dict.has_key(probe):
                return self._dict[probe]
            else:
                DIAError('{0:s} is not recognized as a probe name or type'
                         .format(str(probe)))



    def identify_hotspots(self, **kwargs):
        """Identify probe binding hotspots.

        Keyword arguments are passed to :meth:`set_parameters` method.

        A probe binding hotspot is a grid element with binding free energy
        lower than the value of :attr:`delta_g` parameter and than the
        binding free energies of the surrounding grid elements.

        .. seealso::

            For algorithmic details see :ref:`identifyhotspots`.

        """
        if not self._probes:
            raise DIAError('You need to add probes before you '
                           'can identify binding hotspots.')
        self.set_parameters(**kwargs)
        n_frames = self.parameters['n_frames']
        if n_frames is None:
            raise DIAError('n_frames is a required parameter, and is not set'
                           'to identify binding hotspots.')
        delta_g = self.parameters['delta_g']
        if delta_g is None:
            raise DIAError('delta_g parameter is not set, and is required '
                           'to identify binding hotspots.')
        temperature = self.parameters['temperature']
        # Calculate enrichment for given delta_g
        if delta_g >= 0:
            raise ProbeError('{0:.2f} kcal/mol is not a valid binding '
                                 'free energy limit'.format(delta_g))
        enrichment = convert_dG_to_Kd(-delta_g, temperature)
        self.logger.info('Searching probe binding hotspots with deltaG less '
                         'than {0:.2f} kcal/mol (~{1:.0f} folds enrichment).'
                         .format(delta_g, enrichment))

        start = time.time()
        for probe in self._probes:
            if probe.grid.state == 'original':
                probe.grid.smooth()
            elif probe.grid.state != 'original':
                raise DIAError('{0:s} grid status {1:s} is not recognized'
                               .format(probe.type, probe.grid.state))
        # Get all hotspots from the combined grid (get_hotspots smooths grid)
        if len(self._probes) == 1:
            self._all = self._probes[0]
        else:
            # Merge individual grids
            self._all = Probe(self, 'ALL', np.sum(
                              [probe.grid for probe in self._probes]))
            self._all.grid.filename = self.name + '_ALL' + os.path.splitext(
                                           self._probes[0].grid.filename)[1]
        probes = self._all

        # Calculate population for given delta_g
        population = enrichment * n_frames * probes.expected_population
        #print 'population', population
        #population = population / 5
        which = (probes.grid.array >= population).nonzero()
        #print 'n', len(which[0])
        if len(which[0]) == 0:
            raise ProbeError('No hotspots found with binding free energy '
                             'below {0:.f} kcal/mol (~{0:.0f} folds '
                             'enrichment)'.format(delta_g, enrichment))
        spots = zip(-probes.grid.array[which], which[0], which[1], which[2])
        spots.sort()
        spots = np.array(spots)
        radii = np.array([probe.radius for probe in self._probes])
        charges = np.array([probe.charge for probe in self._probes])
        i = 0
        type_and_occupancy = []
        full_occupancy = []
        charge_list = []
        radii_list = []
        while i < len(spots):
            ijk = tuple(spots[i, 1:].astype('i'))
            counts = [(probe.grid.array[ijk], k)
                      for k, probe in enumerate(self._probes)]
            weights = np.array(counts)[:, 0].T
            weightsum = weights.sum()
            counts.sort(reverse=True)
            counts = np.array(counts)
            counts[:, 0] = counts[:, 0] / weightsum

            full_occupancy.append(counts)
            type_and_occupancy.append((counts[0, 1], counts[0, 0]))

            # eliminate points within weighted radius average
            radius = (radii * weights).sum() / weightsum
            charge_list.append([(charges * weights).sum() / weightsum])
            radii_list.append([radius])
            # an average distance corresponding C-X bond is added
            # to the radius to account for some buffer space
            # when connecting hotspots
            #cutoff = ((radius + 1.4)  / probes.grid.spacing[0]) ** 2
            vdw_compensation = 0
            cutoff = ((radius + vdw_compensation) / probes.grid.spacing[0]) ** 2
            keep = ((spots[:, 1:]-spots[i, 1:]) ** 2).sum(1) > cutoff
            keep[:i+1] = True
            spots = spots[keep]
            i += 1
        spots[:, 0] = -spots[:, 0] / n_frames / probes.expected_population
        spots[:, 0] = -convert_Kd_to_dG(spots[:, 0], temperature)

        self.hotspots = spots


        self.logger.info('{0:d} {1:s} binding spots were identified in '
                         '{2:.2f}s.'.format(len(spots), probes.name,
                         time.time()-start))
        self.logger.info('Minimum binding free energy is {0:.2f} '
                         'kcal/mol.'.format(spots[0,0]))

        # print details of occupancy in debugging mode
        for j, counts in enumerate(full_occupancy):
            self.logger.debug('Hotspot {0:3d} {1:5.2f} kcal/mol {2:s}'.format(
                    j+1, self.hotspots[j, 0],
                    ''.join(
                    [('{0:5.1f}% {1:s} '.format(count[0]*100,
                                            self._probes[int(count[1])].type))
                    for count in counts[(counts[:,0] > 0).nonzero()[0]]]
                    )))


        self.hotspots = np.concatenate([self.hotspots,
                                        np.array(type_and_occupancy)], 1)
        self.hotspots = np.concatenate([self.hotspots,
                                        np.array(charge_list)], 1)
        self.hotspots = np.concatenate([self.hotspots,
                                        np.array(radii_list)], 1)
        # Convert indices to coordinates
        # Example
        #       nX = 10     first = -4       last = 5   spacing = 0.5
        #       0    1    2    3    4    5    6    7    8    9 Python_Index
        #   ----|----|----|----|----|----|----|----|----|----|
        #      -4   -3   -2   -1    0    1    2    3    4    5 Xplor_Index
        # -2.0 -1.5 -1.0 -0.5   0   0.5  1.0  1.5  2.0  2.5 X_coordinate
        # X_coordinate = (Python_Index + first) * spacing
        # Python_Index = (X_coordinate / spacing) - first -> is shifting
        self.hotspots[:, 1:4] = (self.hotspots[:, 1:4] * self._all.grid.spacing
                                 ) + self._all.grid.offset

        # Write a short report for each probe type
        if len(self._probes) > 1:
            for i, probe in enumerate(self._probes):
                which = (self.hotspots[:, 4] == i).nonzero()[0]
                self.logger.info('{0:s}: {1:d} {2:s} binding hotspots '
                                 'were identified.'
                                 .format(probe.type, len(which), probe.name))
                if len(which) > 0:
                    self.logger.info('{0:s}: lowest binding free energy is '
                                     '{1:.2f} kcal/mol.'
                                     .format(probe.type,
                                             self.hotspots[which[0],0]))
        self._resnames = np.array([self._probes[i].type
                                     for i in self.hotspots[:, 4].astype('i')])
        self._atomnames = np.array([self._probes[i].atomname
                                     for i in self.hotspots[:, 4].astype('i')])


    def assess_druggability(self, **kwargs):
        """Identify potential binding sites and estimate achievable affinity.

        Keyword arguments are passed to :meth:`set_parameters` method.

        """
        if self.hotspots is None or not isinstance(self.hotspots, np.ndarray):
            raise DIAError('Binding hotspots have not been identified.')
        self.set_parameters(**kwargs)

        merge_radius = self.parameters.get('merge_radius', None)
        if not isinstance(merge_radius, float):
            raise DIAError('merge_radius of type float must be provided')
        low_affinity = self.parameters.get('low_affinity', None)
        if not isinstance(low_affinity, float):
            raise DIAError('low_affinity of type float must be provided')
        n_probes = self.parameters.get('n_probes', None)
        if not isinstance(n_probes, int):
            raise DIAError('n_probes of type int must be provided')
        min_n_probes = self.parameters.get('min_n_probes', None)
        if not isinstance(min_n_probes, int):
            raise DIAError('min_n_probes of type int must be provided')
        if min_n_probes > n_probes:
            raise DIAError('min_n_probes cannot be greater than n_probes')
        n_solutions = self.parameters.get('n_solutions', None)
        if not isinstance(n_solutions, int):
            raise DIAError('n_solutions of type int must be provided')
        max_charge = self.parameters.get('max_charge', None)
        if not isinstance(max_charge, float):
            raise DIAError('max_charge of type float must be provided')
        n_charged = self.parameters.get('n_charged', None)
        if not isinstance(n_charged, int):
            raise DIAError('n_charged of type int must be provided')

        # Calculate distances
        length = self.hotspots.shape[0]
        dist2 = np.zeros((length, length))
        cutoff2 = merge_radius**2
        coords = self.hotspots[:, 1:4]
        for i in range(length):
            dist2[i+1:, i] = ((coords[i+1:, :] - coords[i])**2).sum(1)

        # Perform single-linkage hierarchical clustering
        start = time.time()
        self.logger.info('Clustering probe binding hotspots.')
        clusters = single_linkage(dist2, cutoff2)
        self.hotspots = np.concatenate([self.hotspots,
                                        clusters.reshape((length, 1))], 1)
        self.logger.info('Clustering completed in {0:.2f}ms.'
                         .format((time.time()-start)*1000))


        # Determine potential sites
        sites = []
        for i in set(clusters):
            which = (self.hotspots[:, -1] == i).nonzero()[0]
            if len(which) >= min_n_probes:
                delta_g = self.hotspots[which, 0].sum()
                #if delta_g <= low_affinity + 5:
                if delta_g < low_affinity:
                    if len(which) <= n_probes:
                        if abs(self.hotspots[which, 6].sum()) > max_charge:
                            continue
                    sites.append((delta_g, len(which), i))
        sites.sort()
        self.logger.info('{0:d} potential sites are identified.'
                         .format(len(sites)))
        #mkp print self.hotspots[0]
        #mkp print sites

        self.logger.info('Calculating achievable affinity ranges.')
        sdict = {}
        for k, (delta_g, size, i) in enumerate(sites):
            if delta_g < low_affinity:
                which = (self.hotspots[:, -1] == i).nonzero()[0]
                hotspots = self.hotspots[which, :]
                mlist = []
                mdict = {}
                n_which = len(which)
                if n_which > n_probes:
                    #mkp print 'len(which) > n_probes'
                    dist2sub = dist2[which, :][:, which]
                    dist2sub = dist2sub + dist2sub.T
                    ###print len(which), 'probes'
                    tempc = 0
                    for i in range(n_which - n_probes):
                        # grow molecules for all hotspots in site except those n_probes with highest dG
                        #mkp print 'for i in range(n_which - n_probes):'
                        grownups = _grow_molecules([], [i], dist2sub,
                                                   cutoff2, n_probes,
                                                   hotspots[:,0])
                        #mkp stop
                        #grownups = _grow_molecules([], [n_which-i-1],
                        #                           dist2sub[:n_which-i, :][:, :n_which-i],
                        #                           cutoff2, n_probes)
                        tempc += len(grownups)
                        #mkp print 'tempc += len(grownups)'
                        for molecule in grownups:
                            mtuple = list(molecule)
                            mtuple.sort()
                            mtuple = tuple(mtuple)
                            if not mdict.has_key(mtuple):
                                mdict[mtuple] = molecule
                                mol_delta_g = hotspots[molecule, 0].sum()
                                mol_charge = hotspots[molecule, 6].sum()
                                mol_n_charged = np.round(np.abs(
                                                  hotspots[molecule, 6])).sum()
                                #if mol_charge != 0:
                                #interact(local=locals())
                                if mol_delta_g <= low_affinity and \
                                   abs(mol_charge) <= max_charge and \
                                   mol_n_charged <= n_charged:
                                    mlist.append((mol_delta_g, list(mtuple),
                                                  mol_charge))
                    ###print tempc, 'grownups'
                    if not mlist:
                        continue
                    mlist.sort()
                    drugindices = []
                    for i in mlist:
                        drugindices.append(i[0])
                    drugindices = np.array(drugindices)
                    affinities = convert_dG_to_Kd(drugindices)
                else:
                    mol_charge = hotspots[:, 6].sum()
                    if abs(mol_charge) <= max_charge:
                        mlist.append((delta_g, range(len(which)), mol_charge))
                    drugindices = np.array([delta_g])
                    affinities = convert_dG_to_Kd(drugindices)
                sdict[k] = {}
                sdict[k]['hotspots'] = hotspots
                sdict[k]['which'] = which
                sdict[k]['size'] = size
                sdict[k]['min_delta_g'] = hotspots[0, 0]
                sdict[k]['mean_delta_g'] = delta_g / size
                sdict[k]['mlist'] = mlist
                sdict[k]['drugindices'] = drugindices
                sdict[k]['affinities'] = affinities

        slist = []
        for key, value in sdict.items():
            slist.append((value['drugindices'][0], key))
        slist.sort()


        for filename in glob(os.path.join(self.workdir,
                                          self.name + '_site_*.pdb')):
            os.remove(filename)

        for k, (temp, key) in enumerate(slist):
            hotspots = sdict[key]['hotspots']
            which = sdict[key]['which']
            atomnames = self._atomnames[which]
            resnames = self._resnames[which]
            # write all hotspots in this site
            filename = os.path.join(self.workdir, self.name +
                                    '_site_{0:d}.pdb'.format(k+1))
            out = open(filename, 'w')
            pdb = get_pdb_lines(hotspots[:, 1:4],
                atomnames=atomnames, resnames=resnames,
                occupancy=hotspots[:, 5], bfactor=hotspots[:, 0])
            out.write(pdb)
            out.close()


            drugindices = sdict[key]['drugindices']
            affinities = sdict[key]['affinities']
            mlist = sdict[key]['mlist']
            self.logger.info('Site {0:d}: {1:d} probe binding hotspots'
                     .format(k+1, sdict[key]['size']))
            self.logger.info('Site {0:d}: Lowest probe binding free energy'
                ' {1:.2f} kcal/mol'.format(k+1, sdict[key]['min_delta_g']))
            self.logger.info('Site {0:d}: Average probe binding free energy'
                '{1:.2f} kcal/mol'.format(k+1, sdict[key]['mean_delta_g']))

            if len(drugindices) > 1:
                self.logger.info('Site {0:d}: Total of {1:d} '
                    'solutions.'.format(k+1, len(drugindices)))

                self.logger.debug('\n' +
                    get_histr(-np.log10(affinities),
                                        label='-log10(affinity)',
                                        title=('Achievable '
                                        'affinities for site {0:d}')
                                        .format(k+1)))
            self.logger.info('Site {0:d}: Lowest drug-like binding '
                             'free energy {1:.2f} kcal/mol'
                             .format(k+1, drugindices[0]))
            self.logger.info('Site {0:d}: Highest drug-like affinity '
                             '{1[0]:.3f} {1[1]:s}'
                     .format(k+1, format_Kd(affinities[0])))


            for m in range(min(n_solutions, len(mlist))):
                which = mlist[m][1]
                mol_charge = mlist[m][2]
                filename = os.path.join(self.workdir, self.name +
                        '_site_{0:d}_soln_{1:d}.pdb'.format(k+1, m+1))
                out = open(filename, 'w')
                pdb = get_pdb_lines(hotspots[which, 1:4],
                                        atomnames=atomnames[which],
                                        resnames=resnames[which],
                                        occupancy=hotspots[which, 5],
                                        bfactor=hotspots[which, 0])
                out.write(pdb)
                out.close()
                self.logger.debug('Site {0:d}: Solution {2:d} binding '
                                 'free energy {1:.2f} kcal/mol'
                                 .format(k+1, drugindices[m], m+1))
                self.logger.debug('Site {0:d}: Solution {2:d} affinity'
                                 ' {1[0]:.3f} {1[1]:s}'.format(k+1,
                                 format_Kd(affinities[m]), m+1))
                self.logger.debug('Site {0:d}: Solution {2:d} total '
                                 'charge {1:.2f} e'
                                 .format(k+1, mol_charge, m+1))
                self.logger.debug('Site {0:d}: Solution {2:d} number of '
                                 'hotspots {1:d}'
                                 .format(k+1, len(which), m+1))

                volume = 0
                volcor = 0
                for a, w in enumerate(which):
                    h1 = hotspots[w]
                    R = h1[7]
                    volume += R ** 3
                    # apply pairwise overlap correction
                    for b in range(a+1, len(which)):
                        h2 = hotspots[which[b]]
                        r = h2[7]
                        d = ((h1[1:4] - h2[1:4]) ** 2).sum() ** 0.5
                        if (R + r) > d:
                            volcor += (np.pi * (d**2 + 2*d*r - 3*r**2 +
                                                2*d*R - 3*R**2 + 6*r*R) *
                                       (R + r - d)**2) / 12 / d
                volume = volume * 4 / 3 * np.pi - volcor
                self.logger.debug('Site {0:d}: Solution {2:d} approximate '
                                 'volume {1:.2f} A^3'
                                 .format(k+1, volume, m+1))





    def write_hotspots(self, probe_type=None):
        """Write probe binding hotspots in PDB format.

        :arg probe_type: probe type to output, if None all probes
        :type probe_type: str or None, default is None

        If :meth:`assess_druggability` method is called, cluster ids of
        hotspots will be printed as the residue number of the hotspot.

        """
        if self.hotspots is None:
            raise DIAError('Binding hotspots have not been identified.')
        which = None
        if probe_type is None:
            which = range(self.hotspots.shape[0])
            filename = os.path.join(self.workdir,
                                    self.name + '_all_hotspots.pdb')
        else:
            for i, probe in enumerate(self._probes):
                if probe.type == probe_type:
                    which = self.hotspots[:, 4] == i
                    break
        if which is None:
            raise DIAError('{0:s} is not recognized as a valid probe type'
                           .format(probe_type))
            filename = os.path.join(self.workdir, self.name + '_' +
                                                  probe_type + '_hotspots.pdb')
        out = open(filename, 'w')
        if probe_type is None:
            out.write('REMARK Hotspots are sorted according to their binding '
            'free energies in ascending order.\n'
            'REMARK Bindiing free energies are written in b-factor column.\n'
            'REMARK Each hotspot is represented by the probe type that is '
            'most frequently observed at its location.\n'
            'REMARK Residue names correspond to so selected probe type.\n'
            'REMARK Occupancy values are the fraction of frames that the '
            'was occupied by the most frequent probe type at that location.\n'
            'REMARK Cluster numbers are written to residue numbers column.\n'
            'REMARK Single-linkage clustering cutoff was {0:.1f}A.\n'
            .format(self.parameters['merge_radius']))
        if self.hotspots.shape[1] > 6:
            resids = self.hotspots[which, -1].flatten().astype('int64') + 1
        else:
            resids = None
        pdb = get_pdb_lines(self.hotspots[:, 1:4],
                                      atomnames=self._atomnames[which],
                                      resnames=self._resnames[which],
                                      resids=resids,
                                      occupancy=self.hotspots[:, 5],
                                      bfactor=self.hotspots[:, 0])
        out.write(pdb)
        out.close()
        self.logger.info('Hotspots are written into file {0:s}.'
                         .format(os.path.relpath(filename)))


    def perform_analysis(self):
        """Perform all analysis steps at once.

        All parameters without a default value must have been set by the user.

        This method runs the following steps:

        * :meth:`identify_hotspots`
        * :meth:`assess_druggability`
        * :meth:`write_hotspots`

        """
        self.identify_hotspots()
        self.assess_druggability()
        self.write_hotspots()


    def evaluate_ligand(self, filename, **kwargs):
        """Predict affinity of a site bound to a given ligand.

        This method selects hotspots within given radius of the ligand atoms
        and calculates affinity based on selected hotspots.

        """
        """
        :arg filename: ligand pdb filename, should contain only ligand atoms
        :type filename: str

        :arg radius: the distance (A) at which hotspots will be considered as
          overlaping with the ligand
        :type radius: float, default is 1.5

        """
        if not self._probes:
            raise DIAError('You need to add probes before you '
                           'can identify binding hotspots.')
        n_frames = self.parameters['n_frames']
        if n_frames is None:
            raise DIAError('n_frames is a required parameter, and is not set'
                           'to identify binding hotspots.')
        delta_g = kwargs.get('delta_g', -0.01)
        if delta_g is None:
            raise DIAError('delta_g parameter is not set, and is required '
                           'to identify binding hotspots.')
        temperature = self.parameters['temperature']
        # Calculate enrichment for given delta_g
        if delta_g >= 0:
            raise ProbeError('{0:.2f} kcal/mol is not a valid binding '
                                 'free energy limit'.format(delta_g))
        enrichment = convert_dG_to_Kd(-delta_g, temperature)
        self.logger.info('Searching probe binding hotspots with deltaG less '
                         'than {0:.2f} kcal/mol (~{1:.0f} folds enrichment).'
                         .format(delta_g, enrichment))

        probes = self._all

        # Calculate population for given delta_g
        population = enrichment * n_frames * probes.expected_population
        #which = (probes.grid.array > population).nonzero()


        self.logger.info('Evaluating binding site of ligand in {0:s}.'.format(
                                                    os.path.relpath(filename)))
        pdb_file = open(filename)
        coords = []
        for line in pdb_file:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                coords.append((float(line[30:38]), float(line[38:46]),
                                    float(line[46:54])))
        pdb_file.close()
        coords = np.array(coords, 'd')
        indices = np.array((coords - self._all.grid.offset)
                            / self._all.grid.spacing, 'i')
        ranges = np.array((kwargs.get('radius', 1.5) / self._all.grid.spacing).round(), 'i')
        ilists = [[], [], []]
        for index in indices:
            for i in range(ranges[0]):
                for j in range(ranges[1]):
                    for k in range(ranges[0]):
                        ilists[0].append(index[0]-i)
                        ilists[0].append(index[0]+i)
                        ilists[1].append(index[1]-j)
                        ilists[1].append(index[1]+j)
                        ilists[2].append(index[2]-k)
                        ilists[2].append(index[2]+k)

        spots = zip(-self._all.grid.array[ilists], ilists[0], ilists[1], ilists[2])
        spots.sort()
        spots = np.array(spots)
        spots = spots[spots[:, 0] < -population, :]

        radii = np.array([probe.radius for probe in self._probes])
        charges = np.array([probe.charge for probe in self._probes])
        i = 0
        type_and_occupancy = []
        full_occupancy = []
        charge_list = []
        radii_list = []
        while i < len(spots):
            ijk = tuple(spots[i, 1:].astype('i'))
            counts = [(probe.grid.array[ijk], k)
                      for k, probe in enumerate(self._probes)]
            weights = np.array(counts)[:, 0].T
            weightsum = weights.sum()
            counts.sort(reverse=True)
            counts = np.array(counts)
            counts[:, 0] = counts[:, 0] / weightsum

            full_occupancy.append(counts)
            type_and_occupancy.append((counts[0, 1], counts[0, 0]))

            # eliminate points within weighted radius average
            radius = (radii * weights).sum() / weightsum
            charge_list.append([(charges * weights).sum() / weightsum])
            radii_list.append([radius])
            # an average distance corresponding C-X bond is added
            # to the radius to account for some buffer space
            # when connecting hotspots
            #cutoff = ((radius + 1.4)  / probes.grid.spacing[0]) ** 2
            cutoff = (radius / probes.grid.spacing[0]) ** 2
            keep = ((spots[:, 1:]-spots[i, 1:]) ** 2).sum(1) > cutoff
            keep[:i+1] = True
            spots = spots[keep]
            i += 1
        spots[:, 0] = -spots[:, 0] / n_frames / probes.expected_population
        spots[:, 0] = -convert_Kd_to_dG(spots[:, 0], temperature)
        hotspots = spots

        # print details of occupancy in debugging mode
        for j, counts in enumerate(full_occupancy):
            self.logger.debug('Hotspot {0:3d} {1:5.2f} kcal/mol {2:s}'.format(
                    j+1, hotspots[j, 0],
                    ''.join(
                    [('{0:5.1f}% {1:s} '.format(count[0]*100,
                                            self._probes[int(count[1])].type))
                    for count in counts[(counts[:,0] > 0).nonzero()[0]]]
                    )))


        hotspots = np.concatenate([hotspots, np.array(type_and_occupancy)], 1)
        hotspots = np.concatenate([hotspots, np.array(charge_list)], 1)
        hotspots = np.concatenate([hotspots, np.array(radii_list)], 1)
        hotspots[:, 1:4] = (hotspots[:, 1:4] * self._all.grid.spacing
                                 ) + self._all.grid.offset
        n_hotspots = len(hotspots)
        self.logger.info('{0:d} hotspots were identified.'
                         .format(n_hotspots))
        resnames = [self._probes[i].type for i in hotspots[:, 4].astype('i')]
        atomnames = [self._probes[i].atomname
                     for i in hotspots[:, 4].astype('i')]
        filename = os.path.join(self.workdir, self.name + '_' +
                                        os.path.split(filename)[-1])
        delta_g = hotspots[:, 0].sum()
        affinity = format_Kd(convert_dG_to_Kd(delta_g))

        self.logger.info('Estimated binding free energy is {0:.2f} kcal/mol.'
                         .format(delta_g))
        self.logger.info('Estimated affinity is {0[0]:.3f} {0[1]:s}.'
                         .format(affinity))

        volume = 0
        volcor = 0
        for a, h1 in enumerate(hotspots):
            R = h1[7]
            volume += R ** 3
            # apply pairwise overlap correction
            for b in range(a+1, n_hotspots):
                h2 = hotspots[b]
                r = h2[7]
                d = ((h1[1:4] - h2[1:4]) ** 2).sum() ** 0.5
                if (R + r) > d:
                    volcor += (np.pi * (d**2 + 2*d*r - 3*r**2 +
                                        2*d*R - 3*R**2 + 6*r*R) *
                               (R + r - d)**2) / 12 / d
        volume = volume * 4 / 3 * np.pi - volcor
        self.logger.info('Solution approximate volume {0:.2f} A^3'
                          .format(volume))

        pdb_file = open(filename, 'w')
        pdb_file.write('REMARK Estimated binding free energy is {0:.2f} '
                       'kcal/mol\n'.format(delta_g))
        pdb_file.write('REMARK Estimated affinity is {0[0]:.3f} {0[1]:s}\n'
                         .format(affinity))
        pdb = get_pdb_lines(hotspots[:, 1:4], atomnames=atomnames,
                                      resnames=resnames,
                                      occupancy=hotspots[:, 5],
                                      bfactor=hotspots[:, 0])
        pdb_file.write(pdb)
        pdb_file.close()
        self.logger.info('Selected hotspots are written into {0:s}.'.format(
                                                    os.path.relpath(filename)))
