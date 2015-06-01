import unittest

class TestImport(unittest.TestCase):
    def test_import(self):
        import vmd

    def test_molecule(self):
        import molecule
        with self.assertRaises(ValueError):
            molecule.load('pdb','bbb')

def run():
    unittest.main()

