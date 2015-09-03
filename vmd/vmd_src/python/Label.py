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
#       $RCSfile: Label.py,v $
#       $Author: johns $        $Locker:  $             $State: Exp $
#       $Revision: 1.4 $        $Date: 2007/01/12 20:19:10 $
#
############################################################################

from VMD import label

class VMDLabel:
	""" VMDLabel represents the state of a label in VMD.  This class should
	not be instantiated directly; use one of the four base classes instead.
	The on/off status is reflected in the 'on' attribute; the value of the
	label (for all but AtomLabel instances, which have no value) is 
	reflected in the 'value' attribute.  Call the update() method to ensure
	that 'on' and 'value' are current."""
	def __init__(self, type, molid, atomid):
		self.type = type
		self.__dict__.update(label.add(self.type, molid, atomid))

	def update(self):
		""" Updates the values of the 'on' and 'value' attributes for this label.
		"""
		self.__dict__.update(label.add(self.type, self.molid, self.atomid))

	def show(self):
		""" Turns this label on."""
		try:
			label.show(self.type, self.__dict__)
		except ValueError:
			raise ValueError, "Label has been deleted."

	def hide(self):
		""" Turns this label off."""
		try:
			label.hide(self.type, self.__dict__)
		except ValueError:
			raise ValueError, "Label has been deleted."

	def delete(self):
		""" Deletes this labe. """ 
		try:
			label.delete(self.type, self.__dict__)
		except ValueError:
			raise ValueError, "Label was already deleted."

	def getValues(self):
		""" Returns a list of the value of this label for all timesteps.  If the
		label's atoms come from different molecules, only the first molecule's
		timesteps will be cycled; the other molecules will use their current
		timesteps."""
		return label.getvalues(self.type, self.__dict__)

class AtomLabel(VMDLabel):
	def __init__(self, m, a):
		VMDLabel.__init__(self, "Atoms", (m,), (a,))

class BondLabel(VMDLabel):
	def __init__(self, m1, m2, a1, a2):
		VMDLabel.__init__(self, "Bonds", (m1, m2), (a1, a2))

class AngleLabel(VMDLabel):
	def __init__(self, m1, m2, m3, a1, a2, a3):
		VMDLabel.__init__(self, "Angles", (m1, m2, m3), (a1, a2, a3))

class DihedralLabel(VMDLabel):
	def __init__(self, m1, m2, m3, m4, a1, a2, a3, a4):
		VMDLabel.__init__(self, "Dihedrals", (m1, m2, m3, m4), (a1, a2, a3, a4))

"""
The following functions return a list of VMDLabel subclass instances 
corresponding to the existing labels in VMD.
"""
def atomLabels():
	""" Returns a list of all existing atom labels."""
	return [apply(AtomLabel, d["molid"]+d["atomid"]) for d in label.listall("Atoms")]

def bondLabels():
	""" Returns a list of all existing bond labels."""
	return [apply(BondLabel, d["molid"]+d["atomid"]) for d in label.listall("Bonds")]

def angleLabels():
	""" Returns a list of all existing angle labels."""
	return [apply(AngleLabel, d["molid"]+d["atomid"]) for d in label.listall("Angles")]

def dihedralLabels():
	""" Returns a list of all existing dihedral labels."""
	return [apply(DihedralLabel, d["molid"]+d["atomid"]) for d in label.listall("Dihedrals")]


if __name__=="__main__":
	from Molecule import *
	from Label import *

	m=Molecule(0)
	b=BondLabel(m, m, 0, 5)
	b2=BondLabel(m, m, 9, 15)
	for i in range(10):
		AtomLabel(m,i)
	
	print atomLabels()
	print bondLabels()
