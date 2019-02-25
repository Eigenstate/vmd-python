"""
Tests the molecule module
"""
import os
import pytest
import numpy as np
from vmd import atomsel, molecule

def test_basic_load(file_3nob):

    assert molecule.num() == 0
    assert molecule.listall() == []

    m = molecule.load("mae", file_3nob)

    assert molecule.num() == 1
    assert molecule.listall() == [m]

    assert molecule.exists(m)
    assert not molecule.exists(m+1)

    assert molecule.name(m) == "3nob.mae"
    with pytest.raises(ValueError):
        molecule.name(m+1)

    molecule.rename(m, "newname")
    assert molecule.name(m) == "newname"
    with pytest.raises(ValueError):
        molecule.rename(name="wrong id", molid=m+1)

    assert molecule.numatoms(molid=m) == 2481
    with pytest.raises(ValueError):
        molecule.numatoms(m+1)

    molecule.delete(m)
    with pytest.raises(ValueError):
        molecule.delete(m)


def test_special_load(file_3nob, file_3frames, file_psf):

    # Graphics type is special
    m = molecule.load("graphics", "")
    assert molecule.numatoms(m) == 0
    molecule.delete(m)

    # Coordinates and data
    m = molecule.load(struct_type="psf", coord_type="pdb",
                      struct_file=file_psf,
                      coord_file=file_3frames)
    assert molecule.numframes(m) == 3

    # Incorrect numbers of arguments
    with pytest.raises(ValueError):
        molecule.load("psf", file_psf, coord_type="pdb")
    with pytest.raises(ValueError):
        molecule.load("psf", file_psf, coord_file=file_3frames)

    # Nonexistent files
    with pytest.raises(RuntimeError):
        molecule.load("psf", "nonexistent_file")

    with pytest.raises(RuntimeError):
        molecule.load("psf", file_psf, "pdb", "nonexistent_file")

    # Wrong number of atoms in coordinate data
    with pytest.raises(RuntimeError):
        x = molecule.load("mae", file_3nob, "pdb", file_3frames)
        print(molecule.get_filenames(x))

    molecule.delete(m)


def test_create():

    m = molecule.new(name="test", natoms=3000)
    assert molecule.name(m) == "test"
    assert molecule.numatoms(m) == 3000

    with pytest.raises(ValueError):
        m = molecule.new("test2", -2000)

    molecule.delete(m)


def test_frames(file_3frames):

    m = molecule.load("pdb", file_3frames)

    assert molecule.numframes(m) == 3
    with pytest.raises(ValueError):
        molecule.numframes(molid=m+1)

    assert molecule.get_frame(m) == 2
    with pytest.raises(ValueError):
        molecule.get_frame(molid=m+1)

    molecule.set_frame(m, frame=2)
    assert molecule.get_frame(molid=m) == 2
    with pytest.raises(ValueError):
        molecule.set_frame(molid=m+1, frame=0)
    with pytest.raises(ValueError):
        molecule.set_frame(frame=20000, molid=m)

    molecule.dupframe(molid=m, frame=0)
    assert molecule.numframes(molid=m) == 4
    molecule.dupframe(m, 0)
    assert molecule.numframes(molid=m) == 5
    with pytest.raises(ValueError):
        molecule.dupframe(m+1)
    with pytest.raises(ValueError):
        molecule.dupframe(m, frame=4000)

    with pytest.warns(DeprecationWarning):
        molecule.delframe(m, beg=4, end=-1)
    assert molecule.numframes(m) == 4
    molecule.delframe(m, first=0, stride=2)
    assert molecule.numframes(m) == 2
    with pytest.raises(ValueError):
        molecule.delframe(m, first=2000)

    molecule.delete(m)


def test_read(file_3frames):

    m = molecule.load("pdb", file_3frames)
    assert molecule.numframes(molid=m) == 3

    with pytest.warns(DeprecationWarning):
        molecule.read(m, "pdb", filename=file_3frames,
                      beg=1, end=1, skip=1, waitfor=-1)
    assert molecule.numframes(m) == 4

    molecule.read(molid=m, filetype="pdb", filename=file_3frames,
                  waitfor=1)
    molecule.cancel(m)
    assert molecule.numframes(m) < 6
    with pytest.raises(ValueError):
        molecule.cancel(molid=m+1)

    molecule.delete(m)


def test_write(tmpdir, file_3frames):

    m = molecule.load("pdb", struct_file=file_3frames)
    tmpdir = str(tmpdir)

    # Deprecated arguments
    with pytest.warns(DeprecationWarning):
        molecule.write(m, "psf", beg=0, end=1,
                       filename=os.path.join(tmpdir, "deprecated.psf"))

    # Write with stride
    molecule.write(molid=m, filetype="pdb", first=0, stride=2,
                   filename=os.path.join(tmpdir, "2frames.pdb"))
    m2 = molecule.load("pdb", os.path.join(tmpdir, "2frames.pdb"))
    assert molecule.numframes(m2) == 2

    # Write a selection
    sel = atomsel("resname ALA", molid=m)
    molecule.write(m, "mae", selection=sel, first=0, last=0,
                   filename=os.path.join(tmpdir, "ala.mae"))
    m3 = molecule.load("mae", os.path.join(tmpdir, "ala.mae"))
    assert molecule.numframes(m3) == 1
    assert set(atomsel(molid=m3).resname) == set(["ALA"])

    # Write an invalid selection on a different molid
    sel2 = atomsel("resname ALA", molid=m3)
    with pytest.raises(ValueError):
        molecule.write(m, "mae", os.path.join(tmpdir, "badsel.mae"),
                       selection=sel2)

    # Write a nonexistent atom selection
    with pytest.raises(TypeError):
        molecule.write(m, "mae", os.path.join(tmpdir, "badsel.mae"),
                       selection=None)

    # Write zero frames
    with pytest.raises(ValueError):
        molecule.write(first=20, last=21, molid=m, filetype="psf",
                       filename=os.path.join(tmpdir, "zeroframes.psf"))

    # Write to an invalid file name (in this case, a directory)
    with pytest.raises(ValueError):
        molecule.write(m, "pdb", filename=os.path.join(tmpdir, "."))

    molecule.delete(m)
    molecule.delete(m2)
    molecule.delete(m3)

@pytest.mark.skip(reason="Cant figure out environment vars")
def test_ssrecalc(file_3nob):

    # No guarantee stride is on this system so set it to true here
    os.environ["STRIDE_BIN"] = "/usr/bin/true"

    m = molecule.load("mae", file_3nob)
    molecule.ssrecalc(m)

    # Fails on zero frames
    molecule.delframe(m)
    with pytest.raises(RuntimeError):
        molecule.ssrecalc(molid=m)

    # Fails on nonexistent molecule
    with pytest.raises(ValueError):
        molecule.ssrecalc(m+1)

    # Fails if STRIDE_BIN is invalid
    del os.environ["STRIDE_BIN"]
    with pytest.raises(RuntimeError):
        molecule.ssrecalc(m)

    molecule.delete(m)


def test_mol_attrs(file_3nob):

    m1 = molecule.load("mae", file_3nob)
    m2 = molecule.load("mae", file_3nob)

    # Get/set top
    assert molecule.get_top() == m2
    molecule.set_top(molid=m1)
    assert molecule.get_top() == m1
    with pytest.raises(ValueError):
        molecule.set_top(m2+1)

    # Get/set visibility
    molecule.set_visible(m1, visible=False)
    assert molecule.get_visible() == False
    assert molecule.get_visible(molid=m2) == True

    with pytest.warns(DeprecationWarning):
        molecule.set_visible(m1, state=True)
    assert molecule.get_visible(molid=m1) == True

    with pytest.raises(ValueError):
        molecule.set_visible(m2+1, True)
    with pytest.raises(TypeError):
        molecule.set_visible(m2, 3)
    with pytest.raises(ValueError):
        molecule.get_visible(m2+1)

    # Get/set periodic
    assert molecule.get_periodic(m2) == {'a': 1.0, 'alpha': 90.0, 'b': 1.0,
                                         'beta': 90.0, 'c': 1.0, 'gamma':
                                         90.0}
    with pytest.raises(ValueError):
        molecule.get_periodic(molid=m1, frame=3000)

    with pytest.raises(ValueError):
        molecule.set_periodic(molid=m2+1, a=2.0)
    with pytest.raises(ValueError):
        molecule.set_periodic(m1, frame=3000, a=20.0)

    molecule.set_periodic(m2, frame=0, a=90.0, b=90.0, c=90.0,
                          alpha=90.0, beta=90.0, gamma=90.0)
    assert list(molecule.get_periodic(m2, frame=0).values()) == [pytest.approx(90.0)]*6
    assert set(molecule.get_periodic(m1, frame=0).values()) != [pytest.approx(90.0)]*6
    molecule.set_periodic(c=20.0)

    assert molecule.get_periodic()["c"] == pytest.approx(20.0)

    molecule.delete(m1)
    molecule.delete(m2)


def test_mol_time(file_3frames):

    m = molecule.load("pdb", file_3frames)
    assert molecule.numframes(m) == 3

    assert molecule.get_physical_time(molid=m, frame=0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        molecule.get_physical_time(m+1)

    # Test all valid combinations of default keyword arguments
    molecule.set_frame(m, 0)
    molecule.set_physical_time(molid=m, frame=2, time=20.0)
    molecule.set_physical_time(m, 10.0, frame=1)
    molecule.set_physical_time(m, 3.0)

    assert molecule.get_physical_time(m) == pytest.approx(3.0)
    molecule.set_frame(m, frame=2)
    assert molecule.get_physical_time(molid=m) == pytest.approx(20.0)
    assert molecule.get_physical_time(m, frame=1) == pytest.approx(10.0)

    with pytest.raises(ValueError):
        molecule.set_physical_time(molid=m+1, time=20.0)
    with pytest.raises(ValueError):
        molecule.set_physical_time(m, time=30.0, frame=3000)

    molecule.delete(m)


def test_mol_descstrs(file_3frames):

    m1 = molecule.load("pdb", file_3frames)
    m2 = molecule.new(name="empty")

    # Filenames
    assert "ala_nma_3frames.pdb" in molecule.get_filenames(molid=m1)[0]
    assert molecule.get_filenames(m2) == []
    with pytest.raises(ValueError):
        molecule.get_filenames(m2+1)

    # File types
    assert molecule.get_filetypes(molid=m1) == ["pdb"]
    assert molecule.get_filetypes(m2) == []
    with pytest.raises(ValueError):
        molecule.get_filetypes(m2+1)

    # Databases
    assert molecule.get_databases(molid=m1) == ["PDB"]
    assert molecule.get_databases(m2) == []
    with pytest.raises(ValueError):
        molecule.get_databases(m2+1)

    # Accessions
    assert molecule.get_accessions(molid=m1) == ["TEST"]
    assert molecule.get_accessions(m2) == []
    with pytest.raises(ValueError):
        molecule.get_accessions(m2+1)

    # Remarks
    assert molecule.get_remarks(molid=m1) == ["REMARK 1   remark 1\nREMARK 2" \
                                             "   also remark 2\n"]
    assert molecule.get_remarks(m2) == []
    with pytest.raises(ValueError):
        molecule.get_remarks(m2+1)

    molecule.delete(m1)
    molecule.delete(m2)

def test_volumetric(file_psf, file_dx):

    # Test read with a volumetric dataset
    m = molecule.load("psf", file_psf)
    molecule.read(m, "dx", file_dx, volsets=[0])

    # Num volumetric
    assert molecule.num_volumetric(m) == 1
    with pytest.raises(ValueError):
        molecule.num_volumetric(m+1)

    # Test with an invalid dataset index, nothing should be read
    molecule.read(m, "dx", file_dx, volsets=[1])
    assert molecule.num_volumetric(m) == 1

    # Get volumetric
    v = molecule.get_volumetric(m, 0)
    assert len(v) == 4
    assert len(v[0]) == 646875
    assert v[1] == (75, 75, 115)
    assert v[2] == pytest.approx((-37.0, -37.0, -57.0))
    assert len(v[3]) == 9
    del v

    with pytest.raises(ValueError):
        molecule.get_volumetric(m+1, 0)
    with pytest.raises(ValueError):
        molecule.get_volumetric(molid=m, dataindex=200)

    # Delete volumetric, deprecated invocation
    with pytest.warns(DeprecationWarning):
        molecule.remove_volumetric(molid=m, dataindex=0)
    assert molecule.num_volumetric(m) == 0

    # Add volumetric
    newvol = list(np.ones((3,6,10)).flatten())

    # separate size keywords are deprecated
    with pytest.warns(DeprecationWarning):
        molecule.add_volumetric(molid=m, name="testvol", xsize=3, ysize=6,
                                zsize=10, data=newvol)
    assert molecule.num_volumetric(m) == 1
    nv = molecule.get_volumetric(m, 0)

    assert nv[0] == newvol
    assert nv[1] == (3, 6, 10)
    assert nv[2] == (0, 0, 0)
    assert nv[3] == tuple(np.identity(3).flatten())

    # Test a reasonable invocation of add with most keywords
    molecule.add_volumetric(m, "testvol", size=(3,6,10), data=newvol,
                            origin=(10,20,-10), xaxis=(0, 1, 0))
    assert molecule.num_volumetric(molid=m) == 2

    nv = molecule.get_volumetric(m, 1)
    assert nv[0] == newvol
    assert nv[1] == (3, 6, 10)
    assert nv[2] == (10, 20, -10)
    assert nv[3][:3] == (0, 1, 0)

    # Invalid size
    with pytest.raises(ValueError):
        molecule.add_volumetric(m, "wrongsize", size=(1, 0, 1), data=[3])

    # Not matching dimensions
    with pytest.raises(ValueError):
        molecule.add_volumetric(m, "wrondim", size=(10, 1, 3), data=newvol)

    # Data not a sequence
    with pytest.raises(TypeError):
        molecule.add_volumetric(m, "wrongtype", size=(2,3,2), data=True)

    # Non-float items in data
    with pytest.raises(TypeError):
        molecule.add_volumetric(m, "wrongdata", size=(1,1,2), data=[3.0, True])

    # Delete correctly
    molecule.delete_volumetric(molid=m, dataindex=1)
    assert molecule.num_volumetric(m) == 1

    # Delete wrong things
    with pytest.raises(ValueError):
        molecule.delete_volumetric(m+1, dataindex=0)

    with pytest.raises(ValueError):
        molecule.delete_volumetric(m, 1)

    molecule.delete(m)
