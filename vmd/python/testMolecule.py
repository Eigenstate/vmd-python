############################################################################
#cr
#cr            (C) Copyright 1995-2007 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################

############################################################################
# RCS INFORMATION:
#
#       $RCSfile: testMolecule.py,v $
#       $Author: johns $        $Locker:  $             $State: Exp $
#       $Revision: 1.16 $        $Date: 2007/01/12 20:19:11 $
#
############################################################################

import unittest
from Molecule import *

class MoleculeTestCase(unittest.TestCase):
  def setUp(self):
    self.mol1 = Molecule()
    self.mol1.load('../proteins/brH.pdb')

  def tearDown(self):
    self.mol1.delete()
    self.mol1 = None

  def testWebPDB(self):
    newmol=Molecule()
    newmol.load('1tit')
    self.failUnlessEqual(newmol.numFrames(), 1)
    newmol.delete()
    
  def testRename(self):
    newname = 'titin from rcsb'
    self.mol1.rename(newname)
    self.failUnlessEqual(self.mol1.name(), newname)

  def testBadID(self):
    thetest=lambda: Molecule(id=-5)
    self.failUnlessRaises(ValueError, thetest)

  def testBadFile(self):
    thetest=lambda: Molecule().load('/path/to/bogus/file','psf')
    self.failUnlessRaises(IOError, thetest)

  def testRepStr(self):
    for rep in self.mol1.reps():
      print rep
  
  def testFiles(self):
    m=Molecule()
    file1='../proteins/alanin.psf'
    file2='../proteins/alanin.dcd'
    m.load(file1)
    m.load(file2)
    self.failUnless(m.files() is not [file1,file2])
    self.failUnless(m.types() is not ['psf', 'dcd'])
    m.delete()

  def testPickle(self):
    import pickle
    m=Molecule()
    file1='../proteins/alanin.psf'
    file2='../proteins/alanin.dcd'
    m.load(file1)
    m.load(file2)
    f=open('test.pkl', 'w')
    pickle.dump(m, f)
    f.close()
    g=open('test.pkl', 'r')
    m2=pickle.load(g)

  def testFrame(self):
    self.mol1.load('../proteins/brH.pdb')
    self.mol1.load('../proteins/brH.pdb')
    self.mol1.load('../proteins/brH.pdb')
    self.failUnlessEqual(self.mol1.setFrame(2).curFrame(), 2)
    
  def testFrameRange(self):
    m=Molecule()
    m.load('../proteins/alanin.psf')
    m.load('../proteins/alanin.dcd', first=10, last=30, step=2, waitfor=-1)
    self.failUnlessEqual(m.numFrames(), 11)
    m.delete()

  def testDelFrame(self):
    m=Molecule()
    m.load('../proteins/alanin.psf')
    m.load('../proteins/alanin.dcd')
    m.delFrame(first=10, last=30, step=2)
    self.failUnlessEqual(m.numFrames(), 89)

  def testDupFrame(self):
    self.mol1.dupFrame(0).dupFrame(1).dupFrame()
    self.failUnlessEqual(self.mol1.numFrames(), 4)

  def testVolsets(self):
    self.mol1.load('/scratch/v46met_solved_2fofc.map',filetype='edm')

  def testRepCreateDestroy(self):
    bonds=MoleculeRep(style="Bonds")
    self.mol1.addRep(bonds)
    self.failUnlessEqual(self.mol1.numReps(), 2)
    self.mol1.delRep(bonds)
    self.failUnlessEqual(self.mol1.numReps(), 1)

  def testClearReps(self):
    self.mol1.addRep(MoleculeRep())
    self.mol1.clearReps()
    self.failUnlessEqual(self.mol1.numReps(), 0)

  def testRepMod(self):
    self.mol1.clearReps()
    rep=MoleculeRep()
    self.mol1.addRep(rep)
    style="VDW"
    color="Index"
    sel="name CA"
    mat="Transparent"
    rep.changeStyle(style)
    rep.changeColor(color)
    rep.changeSelection(sel)
    rep.changeMaterial(mat)
    newrep=self.mol1.reps()[0]
    self.failUnlessEqual(newrep.style, style)
    self.failUnlessEqual(newrep.color, color)
    self.failUnlessEqual(newrep.selection, sel)
    self.failUnlessEqual(newrep.material, mat)
  
  def testRepAutoUpdate(self):
    for rep in self.mol1.reps():
      self.failUnlessEqual(self.mol1.autoUpdate(rep), 0)
      self.mol1.autoUpdate(rep, 1)
      self.failUnlessEqual(self.mol1.autoUpdate(rep), 1)

  def testRepDuplication(self):
    m0a = Molecule(id=self.mol1.id)
    m0b = Molecule(id=self.mol1.id)
    reps=m0a.reps()
    test1=lambda: m0a.addRep(reps[0])
    test2=lambda: m0b.addRep(reps[0])
    self.failUnlessRaises(ValueError, test1)
    self.failUnlessRaises(ValueError, test2)

  def testSSRecalc(self):
    self.failUnless(self.mol1.ssRecalc())

if __name__=="__main__":
  unittest.main()
