#!/usr/bin/env python

from distutils.core import setup, Extension

psfgenfiles=[
   "../src/charmm_file.c",
   "../src/charmm_parse_topo_defs.c",
   "../src/extract_alias.c",
   "../src/hash.c",
   "../src/hasharray.c",
   "../src/memarena.c",
   "../src/pdb_file.c",
   "../src/pdb_file_extract.c",
   "../src/psf_file.c",
   "../src/psf_file_extract.c",
   "../src/psfgen_wrap.c",
   "../src/stringhash.c",
   "../src/topo_defs.c",
   "../src/topo_mol.c",
   "../src/topo_mol_output.c"]

setup(
  name="Psfgen",
  version="1.4.0",
  description="Psf generator",
  author="Justin Gullingsrud and Jim Phillips",
  author_email="vmd@ks.uiuc.edu",
  packages=['Psfgen'],
  ext_modules=[Extension('Psfgen._psfgen', psfgenfiles, include_dirs=['../src'])]
  )

