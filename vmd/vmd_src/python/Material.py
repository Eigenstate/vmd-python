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
#       $RCSfile: Material.py,v $
#       $Author: johns $        $Locker:  $             $State: Exp $
#       $Revision: 1.4 $        $Date: 2007/01/12 20:19:10 $
#
############################################################################

from VMD import material

class Material:
	def __init__(self, name, **keywds):
		""" Create a reference to a material used in VMD.  If the named material
		doesn't exist, a new one will be created with the default properties.
		Pass keyword arguments to set the properties of the newly created 
		or referenced material.  
		"""
		try:
			material.add(name)
		except ValueError:
			pass
		self.__dict__["name"] = name
		apply(material.change, (self.name,), keywds)
	
	def rename(self, newname):
		""" Rename this material.  The name cannot already exist. """
		material.rename(self.name, newname)
		self.name = newname
		return self

	def properties(self):
		""" Returns a list of the setable material properties.  Get/set these
		properties as attributes of the Material instance to change the
		properties of the material."""
		return material.settings(self.name).keys()

	def __getattr__(self, key):
		return material.settings(self.name)[key]

	def __setattr__(self, key, value):
		apply(material.change, (self.name,), {key:value})

	def __str__(self):
		return self.name

def materialList():
	""" Return a list of Material instances for each existing material. """
	return [Material(name) for name in material.listall()]


if __name__=="__main__":
	dull=Material("dull", specular=0)
	dull.ambient = 0.1
	for mat in materialList():
		print mat
