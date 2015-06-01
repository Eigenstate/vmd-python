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
#       $RCSfile: testdisplay.py,v $
#       $Author: johns $        $Locker:  $             $State: Exp $
#       $Revision: 1.4 $        $Date: 2007/01/12 20:19:11 $
#
############################################################################

import unittest
from VMD import display 

class displayTestCase(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testUpdate(self):
    display.update()

  def testUpdateUI(self):
    display.update_ui()

  def testUpdateOnOff(self):
    display.update_off()
    display.update_on()

  def testSetGet(self):
    mydict={
      'eyesep' : 1.0,
      'focallength' : 4.0,
      'height' : 2.0,
      'distance' : 3.0,
      'nearclip' : 1.0,
      'farclip' : 8.0,
      'antialias' : 0,
      'depthcue' : 1,
      'culling' : 1,
      'stereo' : display.stereomodes()[-1],
      'projection' : display.PROJ_PERSP,
      'size' : [400,400]
      }
    display.set(**mydict)
    for key,val in mydict.items():
      self.failUnlessEqual(display.get(key), val)

if __name__=="__main__":
  unittest.main()
