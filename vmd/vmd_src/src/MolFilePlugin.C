/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: MolFilePlugin.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.196 $      $Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   VMD interface to 'molfile' plugins.  Molfile plugins read coordinate
 *   files, structure files, volumetric data, and graphics data.  The data
 *   is loaded into a new or potentially preexisting molecule in VMD.
 *   Some molefile plugins can also export data from VMD to files.
 * 
 * LICENSE:
 *   UIUC Open Source License
 *   http://www.ks.uiuc.edu/Research/vmd/plugins/pluginlicense.html
 * 
 ***************************************************************************/

#include <stdio.h>

#include "MolFilePlugin.h"
#include "Molecule.h"
#include "AtomSel.h"
#include "Timestep.h"
#include "Inform.h"
#include "Scene.h"
#include "molfile_plugin.h"
#include "PeriodicTable.h"
#include "VolumetricData.h"
#include "MoleculeGraphics.h"
#include "QMData.h"
#include "QMTimestep.h"

#if defined(VMDTKCON)
#include "vmdconsole.h"
#endif

#define MINSIZEOF(a, b) \
	((sizeof(a) < sizeof(b)) ? sizeof(a) : sizeof(b))

#define SAFESTRNCPY(a, b) \
	(strncpy(a, b, MINSIZEOF(a, b)))

MolFilePlugin::MolFilePlugin(vmdplugin_t *p)
: plugin((molfile_plugin_t *)p) {
  rv = wv = NULL; 
  numatoms = 0;
  _filename = NULL;
  qm_data = NULL;
#if defined(VMDTKCON)
   plugin->cons_fputs = &vmdcon_fputs;
#else
   plugin->cons_fputs = NULL;
#endif
}

MolFilePlugin::~MolFilePlugin() {
  close();
  delete [] _filename;
}

int MolFilePlugin::read_structure(Molecule *m, int filebonds, int autobonds) {
  if (!rv) return MOLFILE_ERROR; 
  if (!can_read_structure()) return MOLFILE_ERROR;
  if (!m->init_atoms(numatoms)) return MOLFILE_ERROR;

  molfile_atom_t *atoms = 
    (molfile_atom_t *) calloc(1, numatoms*sizeof(molfile_atom_t));

  int rc, i;
  int optflags = MOLFILE_BADOPTIONS; /* plugin must reset this correctly */
  if ((rc = plugin->read_structure(rv, &optflags, atoms))) {
    free(atoms);
    return rc; // propagate error to caller
  }
  if (optflags == int(MOLFILE_BADOPTIONS)) {
    free(atoms);
    msgErr << "MolFilePlugin: plugin didn't initialize optional data flags" << sendmsg;
    msgErr << "MolFilePlugin: file load aborted" << sendmsg;
    return MOLFILE_ERROR; /* abort load, data can't be trusted */
  }


  float *charge = m->charge();
  float *mass = m->mass();
  float *radius = m->radius();
  float *beta = m->beta();
  float *occupancy = m->occupancy();

  // set molecule dataset flags
  if (optflags & MOLFILE_INSERTION)
    m->set_dataset_flag(BaseMolecule::INSERTION);
  if (optflags & MOLFILE_OCCUPANCY)
    m->set_dataset_flag(BaseMolecule::OCCUPANCY);
  if (optflags & MOLFILE_BFACTOR)
    m->set_dataset_flag(BaseMolecule::BFACTOR);
  if (optflags & MOLFILE_MASS)
    m->set_dataset_flag(BaseMolecule::MASS);
  if (optflags & MOLFILE_CHARGE)
    m->set_dataset_flag(BaseMolecule::CHARGE);
  if (optflags & MOLFILE_RADIUS)
    m->set_dataset_flag(BaseMolecule::RADIUS);
  if (optflags & MOLFILE_ALTLOC)
    m->set_dataset_flag(BaseMolecule::ALTLOC);
  if (optflags & MOLFILE_ATOMICNUMBER)
    m->set_dataset_flag(BaseMolecule::ATOMICNUMBER);

  // Force min/max atom radii to be updated since we've loaded new data
  if (optflags & MOLFILE_RADIUS)
    m->set_radii_changed();

  for (i=0; i<numatoms; i++) {
    molfile_atom_t &atom = atoms[i];
    int atomicnumber = (optflags & MOLFILE_ATOMICNUMBER) ? 
      atom.atomicnumber : -1;
    charge[i] = (optflags & MOLFILE_CHARGE) ?
      atom.charge : m->default_charge(atom.name);
 
    // If we're given an explicit mass value, use it.
    // Failing that, try doing a periodic table lookup if we have
    // a valid atomicnumber.  If that fails also, then we use a crude
    // guess based on the atom name string. 
    mass[i] = (optflags & MOLFILE_MASS) ?
      atom.mass : ((atomicnumber > 0) ?
        get_pte_mass(atomicnumber) : m->default_mass(atom.name));

    // If we're given an explicit VDW radius value, use it.
    // Failing that, try doing a periodic table lookup if we have
    // a valid atomicnumber.  If that fails also, then we use a crude
    // guess based on the atom name string. 
    radius[i] = (optflags & MOLFILE_RADIUS) ?
      atom.radius : ((atomicnumber > 0) ?
        get_pte_vdw_radius(atomicnumber) : m->default_radius(atom.name));

    beta[i] = (optflags & MOLFILE_BFACTOR) ?
      atom.bfactor : m->default_beta();
    occupancy[i] = (optflags & MOLFILE_OCCUPANCY) ?
      atom.occupancy : m->default_occup();
    const char *insertion = (optflags & MOLFILE_INSERTION) ? 
      atom.insertion : " ";
    const char *altloc = (optflags & MOLFILE_ALTLOC) ?
      atom.altloc : "";
    if (0 > m->add_atoms(1,
                         atom.name, atom.type, atomicnumber, 
                         atom.resname, atom.resid, 
                         atom.chain, atom.segid, (char *)insertion, altloc)) {
      // if an error occured while adding an atom, we should delete
      // the offending molecule since the data is presumably inconsistent,
      // or at least not representative of what we tried to load
      msgErr << "MolFilePlugin: file load aborted" << sendmsg;
      free(atoms);
      return MOLFILE_ERROR; // abort load, data can't be trusted
    }
  }
  free(atoms);

  if (filebonds && can_read_bonds()) {
    int nbonds, *from, *to, nbondtypes, *bondtype;
    float *bondorder;
    char **bondtypename;

    // must explicitly set these since they may are otherwise only 
    // initialized by the read_bonds() call in the new ABI
    nbondtypes = 0;
    bondtype = NULL;
    bondtypename = NULL;
 
#if vmdplugin_ABIVERSION >= 15
    if (plugin->read_bonds(rv, &nbonds, &from, &to, &bondorder, &bondtype, 
                           &nbondtypes, &bondtypename)) 
#else
    if (plugin->read_bonds(rv, &nbonds, &from, &to, &bondorder)) 
#endif
    {
      
      msgErr << "Error reading bond information." << sendmsg;
      if (autobonds)
        m->find_bonds_from_timestep();
    } else {
      // if we didn't get an error, but the plugin didn't define bonds
      // for some reason (i.e. no usable PDB CONECT records found)
      // fall back to automatic bond search
      if (nbonds == 0 && from == NULL && to == NULL) {
        if (autobonds)
          m->find_bonds_from_timestep();
      } else if (nbonds > 0) {
        // insert bondtype names into our pool of names. this will preserve 
        // the indexing if no other names do already exist.
        if (bondtypename != NULL)
          for (i=0; i < nbondtypes; i++)
            m->bondTypeNames.add_name(bondtypename[i], i);

        // If the bonds provided by the plugin were only the for
        // non-standard residues (i.e. PDB CONECT records) then we'll
        // also do our regular bond search, combining all together, but 
        // preventing duplicates
        if (optflags & MOLFILE_BONDSSPECIAL) {
          if (autobonds) {
            m->find_unique_bonds_from_timestep(); // checking for duplicates
          }

          // indicate what kind of data will be available.
          m->set_dataset_flag(BaseMolecule::BONDS);
          if (bondorder != NULL) 
            m->set_dataset_flag(BaseMolecule::BONDORDERS);
          if (bondtype  != NULL) 
            m->set_dataset_flag(BaseMolecule::BONDTYPES);
          
          // Now add in all of the special bonds provided by the plugin
          for (i=0; i<nbonds; i++) {
            float bo =  1;
            int   bt = -1;
            if (bondorder != NULL) 
              bo = bondorder[i];
            if (bondtype  != NULL) {
              if (bondtypename != NULL) {
                bt = m->bondTypeNames.typecode(bondtypename[bondtype[i]]);
              } else {
                bt = bondtype[i];
              }
            }
            m->add_bond_dupcheck(from[i]-1, to[i]-1, bo, bt); 
          }
        } else {

          // indicate what kind of data will be available...
          m->set_dataset_flag(BaseMolecule::BONDS);
          if (bondorder != NULL) 
            m->set_dataset_flag(BaseMolecule::BONDORDERS);
          if (bondtype  != NULL) 
            m->set_dataset_flag(BaseMolecule::BONDTYPES);

          // ...and add the bonds.
          for (i=0; i<nbonds; i++) {
            float bo =  1;
            int   bt = -1;
            if (bondorder != NULL) bo = bondorder[i];
            if (bondtype  != NULL) bt = bondtype[i];
            m->add_bond(from[i]-1, to[i]-1, bo, bt); 
          }
        }
      }
    }
  } else {
    if (autobonds)
      m->find_bonds_from_timestep();
  }

  // if the plugin can read angles, dihedrals, impropers, cross-terms, 
  // do it here...
  if (can_read_angles()) {
    int numangles, *angles, *angletypes, numangletypes;
    int numdihedrals, *dihedrals, *dihedraltypes, numdihedraltypes;
    int numimpropers, *impropers, *impropertypes, numimpropertypes; 
    int numcterms, *cterms, ctermcols, ctermrows;
    char **angletypenames, **dihedraltypenames, **impropertypenames;

    // initialize to empty so that the rest of the code works well.
    angletypes = dihedraltypes = impropertypes = NULL;
    numangletypes = numdihedraltypes = numimpropertypes = 0;
    angletypenames = dihedraltypenames = impropertypenames = NULL;
    
#if vmdplugin_ABIVERSION >= 16
    if (plugin->read_angles(rv, &numangles, &angles, &angletypes, 
                            &numangletypes, &angletypenames, &numdihedrals,
                            &dihedrals,  &dihedraltypes, &numdihedraltypes, 
                            &dihedraltypenames, &numimpropers, &impropers,
                            &impropertypes, &numimpropertypes, 
                            &impropertypenames, &numcterms, &cterms, 
                            &ctermcols, &ctermrows))
      
#else
    double *angleforces, *dihedralforces, *improperforces, *ctermforces;
    angleforces = dihedralforces = improperforces = ctermforces = NULL;
    if (plugin->read_angles(rv, 
                            &numangles, &angles, &angleforces,
                            &numdihedrals, &dihedrals, &dihedralforces,
                            &numimpropers, &impropers, &improperforces,
                            &numcterms, &cterms, &ctermcols, &ctermrows, 
                            &ctermforces)) 
#endif
    {
      msgErr << "Error reading angle and cross-term information." << sendmsg;
    } else {
      // we consider angles, dihedrals and impropers all as "angles".
      // it is hard to come up with a scenario where something else
      // as an all-or-nothing setup would happen.
      if ( (angletypes != NULL) || (dihedraltypes != NULL) 
           || (impropertypes != NULL) )
        m->set_dataset_flag(BaseMolecule::ANGLETYPES);

      // insert type names into our pool of names. this will preserve 
      // the indexing if no other names do already exist.
      if (angletypenames != NULL)
        for (i=0; i < numangletypes; i++)
          m->angleTypeNames.add_name(angletypenames[i], i);

      if (dihedraltypenames != NULL)
        for (i=0; i < numdihedraltypes; i++)
          m->dihedralTypeNames.add_name(dihedraltypenames[i], i);

      if (impropertypenames != NULL)
        for (i=0; i < numimpropertypes; i++)
          m->improperTypeNames.add_name(impropertypenames[i], i);

      // XXX: using a similar interface as with bonds would become
      //      quite complicated. not sure if it is really needed.
      //      for the moment we forgo it and force a simple all-or-nothing
      //      scheme instead. incrementally adding and removing of
      //      angles has to be done from (topology building) scripts.
      if (numangles > 0 || numdihedrals > 0 || numimpropers > 0) {
        m->set_dataset_flag(BaseMolecule::ANGLES);
        m->clear_angles();
        if (angletypes != NULL) {
          int type;

          for (i=0; i<numangles; i++) {
            type = angletypes[i];
            if (angletypenames != NULL)
              type = m->angleTypeNames.typecode(angletypenames[type]);

            m->add_angle(angles[3L*i]-1, angles[3L*i+1]-1, angles[3L*i+2]-1, type);
          }
        } else {
          for (i=0; i<numangles; i++) {
            m->add_angle(angles[3L*i]-1, angles[3L*i+1]-1, angles[3L*i+2]-1);
          }
        }

        m->clear_dihedrals();
        if (dihedraltypes != NULL) {
          int type;

          for (i=0; i<numdihedrals; i++) {
            type = dihedraltypes[i];
            if (dihedraltypenames != NULL)
              type = m->dihedralTypeNames.typecode(dihedraltypenames[type]);

            m->add_dihedral(dihedrals[4L*i]-1,   dihedrals[4L*i+1]-1, 
                            dihedrals[4L*i+2]-1, dihedrals[4L*i+3]-1, type);
          }
        } else {
          for (i=0; i<numdihedrals; i++) {
            m->add_dihedral(dihedrals[4L*i]-1,   dihedrals[4L*i+1]-1, 
                            dihedrals[4L*i+2]-1, dihedrals[4L*i+3]-1);
          }
        }
        
        m->clear_impropers();
        if (impropertypes != NULL) {
          int type;

          for (i=0; i<numimpropers; i++) {
            type = impropertypes[i];
            if (impropertypenames != NULL)
              type = m->improperTypeNames.typecode(impropertypenames[type]);

            m->add_improper(impropers[4L*i]-1,   impropers[4L*i+1]-1, 
                            impropers[4L*i+2]-1, impropers[4L*i+3]-1, type);
          }
        } else {
          for (i=0; i<numimpropers; i++) {
            m->add_improper(impropers[4L*i]-1,   impropers[4L*i+1]-1, 
                            impropers[4L*i+2]-1, impropers[4L*i+3]-1);
          }
        }
      }

      // cross terms have no types. they are a CHARMM-only thing.
      if (numcterms > 0) {
        m->set_dataset_flag(BaseMolecule::CTERMS);
        for (i=0; i<numcterms; i++)
          m->add_cterm(cterms[8L*i]-1,   cterms[8L*i+1]-1, cterms[8L*i+2]-1,
                       cterms[8L*i+3]-1, cterms[8L*i+4]-1, cterms[8L*i+5]-1,
                       cterms[8L*i+6]-1, cterms[8L*i+7]-1);
      }
    }
  }

  return MOLFILE_SUCCESS;
}

  
int MolFilePlugin::read_optional_structure(Molecule *m, int filebonds) {
  if (!rv) return MOLFILE_ERROR; 
  if (!can_read_structure()) return MOLFILE_ERROR;
  if (numatoms != m->nAtoms) return MOLFILE_ERROR;

  molfile_atom_t *atoms = (molfile_atom_t *) calloc(1, numatoms*sizeof(molfile_atom_t));

  int rc, i;
  int optflags = MOLFILE_BADOPTIONS; /* plugin must reset this correctly */
  if ((rc = plugin->read_structure(rv, &optflags, atoms))) {
    free(atoms);
    return rc; // propagate error to caller
  }
  if (optflags == int(MOLFILE_BADOPTIONS)) {
    free(atoms);
    msgErr << "MolFilePlugin: plugin didn't initialize optional data flags" << sendmsg;
    msgErr << "MolFilePlugin: file load aborted" << sendmsg;
    return MOLFILE_ERROR; /* abort load, data can't be trusted */
  }

  // set molecule dataset flags
  if (optflags & MOLFILE_OCCUPANCY)
    m->set_dataset_flag(BaseMolecule::OCCUPANCY);
  if (optflags & MOLFILE_BFACTOR)
    m->set_dataset_flag(BaseMolecule::BFACTOR);
  if (optflags & MOLFILE_MASS)
    m->set_dataset_flag(BaseMolecule::MASS);
  if (optflags & MOLFILE_CHARGE)
    m->set_dataset_flag(BaseMolecule::CHARGE);
  if (optflags & MOLFILE_RADIUS)
    m->set_dataset_flag(BaseMolecule::RADIUS);
  if (optflags & MOLFILE_ATOMICNUMBER)
    m->set_dataset_flag(BaseMolecule::ATOMICNUMBER);

  // Force min/max atom radii to be updated since we've loaded new data
  if (optflags & MOLFILE_RADIUS)
    m->set_radii_changed();

  float *charge = m->charge();
  float *mass = m->mass();
  float *radius = m->radius();
  float *beta = m->beta();
  float *occupancy = m->occupancy();
  for (i=0; i<numatoms; i++) {
    if (optflags & MOLFILE_OCCUPANCY) {
      occupancy[i] = atoms[i].occupancy;
    }

    if (optflags & MOLFILE_BFACTOR) {
      beta[i] = atoms[i].bfactor;
    }

    if (optflags & MOLFILE_MASS) {
      mass[i] = atoms[i].mass;
    }

    if (optflags & MOLFILE_CHARGE) {
      charge[i] = atoms[i].charge;
    }

    if (optflags & MOLFILE_RADIUS) {
      radius[i] = atoms[i].radius;
    }

    if (optflags & MOLFILE_ATOMICNUMBER) {
      m->atom(i)->atomicnumber = atoms[i].atomicnumber;
    }
  }
  free(atoms);

  // if no bonds are added, then we do not re-analyze the structure
  if (!can_read_bonds()) 
    return MOLFILE_SUCCESS;

  // When tacking on trajectory frames from PDB files with CONECT records,
  // we have to prevent partial CONECT record bonding information from
  // blowing away existing connectivity information derived from 
  // automatic bond search results or from a previously loaded file with
  // complete bond information such as a PSF.  We only accept 
  // complete bonding connectivity updates, no partial updates are allowed.
  // Also bonding information updates can be disabled by the user with the
  // filebonds flag.
  if (!(optflags & MOLFILE_BONDSSPECIAL) && filebonds) {
    int nbonds, *from, *to, nbondtypes, *bondtype;
    float *bondorder;
    char **bondtypename;
  
    // must explicitly set these since they may are otherwise only
    // initialized by the read_bonds() call in the new ABI
    nbondtypes = 0;
    bondtype = NULL;
    bondtypename = NULL;

#if vmdplugin_ABIVERSION >= 15
    if (plugin->read_bonds(rv, &nbonds, &from, &to, &bondorder, &bondtype, 
                           &nbondtypes, &bondtypename)) 
      return MOLFILE_SUCCESS;
#else
    if (plugin->read_bonds(rv, &nbonds, &from, &to, &bondorder)) 
      return MOLFILE_SUCCESS;
#endif

    // insert bondtype names into our pool of names. this will preserve 
    // the indexing if no other names do already exist.
    if (bondtypename != NULL)
      for (i=0; i < nbondtypes; i++)
        m->bondTypeNames.add_name(bondtypename[i], i);
      
    if (nbonds == 0) 
      return MOLFILE_SUCCESS;

    for (i=0; i<numatoms; i++) 
      m->atom(i)->bonds = 0;

    m->set_dataset_flag(BaseMolecule::BONDS);
    if (bondorder != NULL)
      m->set_dataset_flag(BaseMolecule::BONDORDERS);
    if (bondtype != NULL)
      m->set_dataset_flag(BaseMolecule::BONDTYPES);

    for (i=0; i<nbonds; i++) {
      float bo = 1;
      int bt = -1;

      if (bondorder != NULL) 
        bo = bondorder[i];
      if (bondtype != NULL) {
        if (bondtypename != NULL) {
          bt = m->bondTypeNames.typecode(bondtypename[bondtype[i]]);
        } else {
          bt = bondtype[i];
        }
      }
        
      m->add_bond(from[i]-1, to[i]-1, bo, bt); // real bond order
    } 
  } else {
    // if no bonds are added, then we do not re-analyze the structure
    return MOLFILE_SUCCESS;
  }

  // if the plugin can read angles, dihedrals, impropers, cross-terms, 
  // do it here...
  if (can_read_angles()) {
    int numangles, *angles, *angletypes, numangletypes;
    int numdihedrals, *dihedrals, *dihedraltypes, numdihedraltypes;
    int numimpropers, *impropers, *impropertypes, numimpropertypes; 
    int numcterms, *cterms, ctermcols, ctermrows;
    char **angletypenames, **dihedraltypenames, **impropertypenames;

    // initialize to empty so that the rest of the code works well.
    angletypes = dihedraltypes = impropertypes = NULL;
    numangletypes = numdihedraltypes = numimpropertypes = 0;
    angletypenames = dihedraltypenames = impropertypenames = NULL;
    
#if vmdplugin_ABIVERSION >= 16
    if (plugin->read_angles(rv, &numangles, &angles, &angletypes, 
                            &numangletypes, &angletypenames, &numdihedrals,
                            &dihedrals,  &dihedraltypes, &numdihedraltypes, 
                            &dihedraltypenames, &numimpropers, &impropers,
                            &impropertypes, &numimpropertypes, 
                            &impropertypenames, &numcterms, &cterms, 
                            &ctermcols, &ctermrows))
      
#else
    double *angleforces, *dihedralforces, *improperforces, *ctermforces;
    angleforces = dihedralforces = improperforces = ctermforces = NULL;
    if (plugin->read_angles(rv, 
                            &numangles, &angles, &angleforces,
                            &numdihedrals, &dihedrals, &dihedralforces,
                            &numimpropers, &impropers, &improperforces,
                            &numcterms, &cterms, &ctermcols, &ctermrows, 
                            &ctermforces)) 
#endif
    {
      msgErr << "Error reading angle and cross-term information." << sendmsg;
    } else {
      // we consider angles, dihedrals and impropers all as "angles".
      // it is hard to come up with a scenario where something else
      // as an all-or-nothing setup would happen.
      if ( (angletypes != NULL) || (dihedraltypes != NULL) 
           || (impropertypes != NULL) )
        m->set_dataset_flag(BaseMolecule::ANGLETYPES);

      // insert type names into our pool of names. this will preserve 
      // the indexing if no other names do already exist.
      if (angletypenames != NULL)
        for (i=0; i < numangletypes; i++)
          m->angleTypeNames.add_name(angletypenames[i], i);

      if (dihedraltypenames != NULL)
        for (i=0; i < numdihedraltypes; i++)
          m->dihedralTypeNames.add_name(dihedraltypenames[i], i);

      if (impropertypenames != NULL)
        for (i=0; i < numimpropertypes; i++)
          m->improperTypeNames.add_name(impropertypenames[i], i);

      // XXX: using a similar interface as with bonds would become
      //      quite complicated. not sure if it is really needed.
      //      for the moment we force a simple all-or-nothing
      //      scheme and only allow incrementally adding and removing
      //      angles from (topology building) scripts.
      if (numangles > 0 || numdihedrals > 0 || numimpropers > 0) {
        m->set_dataset_flag(BaseMolecule::ANGLES);
        m->clear_angles();
        if (angletypes != NULL) {
          int type;

          for (i=0; i<numangles; i++) {
            type = angletypes[i];
            if (angletypenames != NULL)
              type = m->angleTypeNames.typecode(angletypenames[type]);

            m->add_angle(angles[3L*i]-1, angles[3L*i+1]-1, angles[3L*i+2]-1, type);
          }
        } else {
          for (i=0; i<numangles; i++) {
            m->add_angle(angles[3L*i]-1, angles[3L*i+1]-1, angles[3L*i+2]-1);
          }
        }

        m->clear_dihedrals();
        if (dihedraltypes != NULL) {
          int type;

          for (i=0; i<numdihedrals; i++) {
            type = dihedraltypes[i];
            if (dihedraltypenames != NULL)
              type = m->dihedralTypeNames.typecode(dihedraltypenames[type]);

            m->add_dihedral(dihedrals[4L*i]-1,   dihedrals[4L*i+1]-1, 
                            dihedrals[4L*i+2]-1, dihedrals[4L*i+3]-1, type);
          }
        } else {
          for (i=0; i<numdihedrals; i++) {
            m->add_dihedral(dihedrals[4L*i]-1,   dihedrals[4L*i+1]-1, 
                            dihedrals[4L*i+2]-1, dihedrals[4L*i+3]-1);
          }
        }
        
        m->clear_impropers();
        if (impropertypes != NULL) {
          int type;

          for (i=0; i<numimpropers; i++) {
            type = impropertypes[i];
            if (impropertypenames != NULL)
              type = m->improperTypeNames.typecode(impropertypenames[type]);

            m->add_improper(impropers[4L*i]-1,   impropers[4L*i+1]-1, 
                            impropers[4L*i+2]-1, impropers[4L*i+3]-1, type);
          }
        } else {
          for (i=0; i<numimpropers; i++) {
            m->add_improper(impropers[4L*i]-1,   impropers[4L*i+1]-1, 
                            impropers[4L*i+2]-1, impropers[4L*i+3]-1);
          }
        }
      }

      // cross terms have no types. they are a CHARMM-only thing.
      if (numcterms > 0) {
        m->set_dataset_flag(BaseMolecule::CTERMS);
        for (i=0; i<numcterms; i++)
          m->add_cterm(cterms[8L*i]-1,   cterms[8L*i+1]-1, cterms[8L*i+2]-1,
                       cterms[8L*i+3]-1, cterms[8L*i+4]-1, cterms[8L*i+5]-1,
                       cterms[8L*i+6]-1, cterms[8L*i+7]-1);
      }
    }
  }

  // (re)analyze the molecular structure, since bonds/angles/etc
  // may have been changed
  m->analyze();

  // force all reps and selections to be recalculated
  m->force_recalc(DrawMolItem::COL_REGEN | DrawMolItem::SEL_REGEN);

  // force secondary structure to be recalculated
  m->invalidate_ss(); 
 
  return MOLFILE_SUCCESS;
}

void MolFilePlugin::set_natoms(int n) {
  numatoms = n;
}

int MolFilePlugin::init_read(const char *file) {
  rv = NULL;
  if (can_read_structure() || can_read_timesteps() || can_read_graphics() ||
      can_read_volumetric() || can_read_metadata() || can_read_qm()) { 
    rv = plugin->open_file_read(file, plugin->name, &numatoms);
  }
  if (!rv) return MOLFILE_ERROR;
  delete [] _filename;
  _filename = stringdup(file);
  return MOLFILE_SUCCESS;
}


int MolFilePlugin::read_timestep_pagealign_size(void) {
#if vmdplugin_ABIVERSION > 17
  if (!rv) return 1;
  if (can_read_pagealigned_timesteps()) {
    int sz;
    plugin->read_timestep_pagealign_size(rv, &sz);

    // If the page alignment size passes sanity check, we use it...
    if ((sz != 1) && 
        ((sz < MOLFILE_DIRECTIO_MIN_BLOCK_SIZE) &&
         (sz > MOLFILE_DIRECTIO_MAX_BLOCK_SIZE))) {
      msgWarn << "Plugin returned bad page alignment size!: " 
              << sz << sendmsg;
    }

    if (getenv("VMDPLUGINVERBOSE") != NULL) {
      msgInfo << "Plugin returned page alignment size: " << sz << sendmsg;
    }

    return sz;
  } else {
    if (getenv("VMDPLUGINVERBOSE") != NULL) {
      msgInfo << "Plugin can't read page aligned timesteps." << sendmsg;
    }
  }

  return 1;
#else
#if 1
  return 1; // assume non-blocked I/O
#else
  // Enable VMD to cope with hard-coded revs of jsplugin if we want
  return MOLFILE_DIRECTIO_MAX_BLOCK_SIZE;
#endif
#endif
}


Timestep *MolFilePlugin::next(Molecule *m, int ts_pagealign_sz) {
  if (!rv) return NULL;
  if (numatoms <= 0) return NULL;
  if (!(can_read_timesteps() || can_read_qm_timestep())) return NULL;
  molfile_timestep_t timestep;
  memset(&timestep, 0, sizeof(molfile_timestep_t));

  // allocate space for velocities only if 
  // 1) the plugin implements read_timestep_metadata;
  // 2) metadata->has_velocities is TRUE.
  float *velocities = NULL;

  // XXX this needs to be pulled out into the read init code
  // rather than being done once per timestep
  molfile_timestep_metadata_t meta;
  if (can_read_timestep_metadata()) {
    memset(&meta, 0, sizeof(molfile_timestep_metadata));
    plugin->read_timestep_metadata(rv, &meta);
    if (meta.has_velocities) {
      velocities = new float[3L*numatoms];
    }
  }

  // QM timestep metadata can be different on every single timestep
  // so we need to query it prior to reading the timestep so we can 
  // allocate the right buffer sizes etc. 
  molfile_qm_timestep_metadata_t qmmeta;
  if (can_read_qm_timestep_metadata()) {
    memset(&qmmeta, 0, sizeof(molfile_qm_timestep_metadata));
    // XXX need to add timestep parameter or other method to specify
    //     which frame this applies to, else keep it the way it is
    //     and rename the plugin function appropriately.
    plugin->read_qm_timestep_metadata(rv, &qmmeta);
  }

  // set useful defaults for unit cell information
  // a non-periodic structure has cell lengths of zero
  // if it's periodic in less than 3 dimensions, then only the
  // periodic directions will be non-zero.
  timestep.A = timestep.B = timestep.C = 0.0f;

  // cells are rectangular until told otherwise
  timestep.alpha = timestep.beta = timestep.gamma = 90.0f;

  Timestep *ts = new Timestep(numatoms, ts_pagealign_sz);
  timestep.coords = ts->pos; 
  timestep.velocities = velocities;
  ts->vel = velocities;
 
  int rc = 0;
  molfile_qm_metadata_t *qm_metadata = NULL; // this just a dummy
  molfile_qm_timestep_t qm_timestep;
  memset(&qm_timestep, 0, sizeof(molfile_qm_timestep_t));

  // XXX this needs to be fixed so that a file format that can 
  //     optionally contain either QM or non-QM data will work correctly
  if (can_read_qm_timestep()) {
    qm_timestep.scfenergies = new double[qmmeta.num_scfiter];
    qm_timestep.wave = new molfile_qm_wavefunction_t[qmmeta.num_wavef];
    memset(qm_timestep.wave, 0, qmmeta.num_wavef*sizeof(molfile_qm_wavefunction_t));
    int i;
    for (i=0; (i<MOLFILE_MAXWAVEPERTS && i<qmmeta.num_wavef); i++) {
      qm_timestep.wave[i].wave_coeffs = 
        new float[qmmeta.num_orbitals_per_wavef[i]*qmmeta.wavef_size];
      if (qmmeta.has_orben_per_wavef[i]) {
        qm_timestep.wave[i].orbital_energies =
          new float[qmmeta.num_orbitals_per_wavef[i]];
      }
      if (qmmeta.has_occup_per_wavef[i]) {
        qm_timestep.wave[i].occupancies = 
          new float[qmmeta.num_orbitals_per_wavef[i]];
      }
    }
    if (qmmeta.has_gradient) {
      qm_timestep.gradient = new float[3L*numatoms];
    }
    if (qmmeta.num_charge_sets) {
      qm_timestep.charges = new double[numatoms*qmmeta.num_charge_sets];
      qm_timestep.charge_types = new int[qmmeta.num_charge_sets];
    }
    rc = plugin->read_timestep(rv, numatoms, &timestep, qm_metadata, &qm_timestep);
  } else {
    rc = plugin->read_next_timestep(rv, numatoms, &timestep);
  }

  if (rc) {
    if (can_read_qm_timestep()) {
      delete [] qm_timestep.scfenergies;
      if (qm_timestep.gradient) delete [] qm_timestep.gradient;
      if (qm_timestep.charges)  delete [] qm_timestep.charges;
      if (qm_timestep.charge_types) delete [] qm_timestep.charge_types;
      int i;
      for (i=0; i<qmmeta.num_wavef; i++) {
        delete [] qm_timestep.wave[i].wave_coeffs;
        delete [] qm_timestep.wave[i].orbital_energies;
      }
      delete [] qm_timestep.wave;
    }
    delete ts;
    ts = NULL; 
  } else {
    ts->a_length = timestep.A;
    ts->b_length = timestep.B;
    ts->c_length = timestep.C;
    ts->alpha = timestep.alpha;
    ts->beta = timestep.beta;
    ts->gamma = timestep.gamma;
    ts->physical_time = timestep.physical_time;
    if (can_read_qm_timestep()) {
      int i;
      int *chargetypes = new int[qmmeta.num_charge_sets];
      for (i=0; i<qmmeta.num_charge_sets; i++) {
        switch (qm_timestep.charge_types[i]) {
        case  MOLFILE_QMCHARGE_MULLIKEN:
          chargetypes[i] = QMCHARGE_MULLIKEN;
          break;
        case MOLFILE_QMCHARGE_LOWDIN:
          chargetypes[i] = QMCHARGE_LOWDIN;
          break;
        case MOLFILE_QMCHARGE_ESP:
          chargetypes[i] = QMCHARGE_ESP;
          break;
        case MOLFILE_QMCHARGE_NPA:
          chargetypes[i] = QMCHARGE_NPA;
          break;
        default:
          chargetypes[i] = QMCHARGE_UNKNOWN;
        }
      }

      ts->qm_timestep = new QMTimestep(numatoms);
      ts->qm_timestep->set_charges(qm_timestep.charges,
                                   chargetypes,
                                   numatoms, qmmeta.num_charge_sets);
      delete [] chargetypes;

      ts->qm_timestep->set_scfenergies(qm_timestep.scfenergies,
                                       qmmeta.num_scfiter);


      // signa_ts is the list of numsigts signatures for the wavefunctions
      // already processed for this timestep. 
      int num_signa_ts = 0;
      wavef_signa_t *signa_ts = NULL;
      for (i=0; i<qmmeta.num_wavef; i++) {
        // We need to translate between the macros used in the plugins
        // and the one ones in VMD, here so that they can be independent
        // and we don't have to include molfile_plugin.h anywhere else
        // in VMD.
        int wavef_type;
        switch (qm_timestep.wave[i].type) {
        case MOLFILE_WAVE_CANON:
          wavef_type = WAVE_CANON;
          break;
        case MOLFILE_WAVE_CINATUR:
          wavef_type = WAVE_CINATUR;
          break;
        case MOLFILE_WAVE_GEMINAL:
          wavef_type = WAVE_GEMINAL;
          break;
        case MOLFILE_WAVE_BOYS:
          wavef_type = WAVE_BOYS;
          break;
        case MOLFILE_WAVE_RUEDEN:
          wavef_type = WAVE_RUEDEN;
          break;
        case MOLFILE_WAVE_PIPEK:
          wavef_type = WAVE_PIPEK;
          break;
        case MOLFILE_WAVE_MCSCFOPT:
          wavef_type = WAVE_MCSCFOPT;
          break;
        case MOLFILE_WAVE_MCSCFNAT:
          wavef_type = WAVE_MCSCFNAT;
          break;
        default:
          wavef_type = WAVE_UNKNOWN;
        }

        // Add new wavefunction to QM timestep
        ts->qm_timestep->add_wavefunction(qm_data,
                           qmmeta.wavef_size,
                           qmmeta.num_orbitals_per_wavef[i],
                           qm_timestep.wave[i].wave_coeffs,
                           qm_timestep.wave[i].orbital_energies,
                           qm_timestep.wave[i].occupancies,
                           qm_timestep.wave[i].orbital_ids,
                           qm_timestep.wave[i].energy,
                           wavef_type,
                           qm_timestep.wave[i].spin,
                           qm_timestep.wave[i].excitation,
                           qm_timestep.wave[i].multiplicity,
                           qm_timestep.wave[i].info,
                           signa_ts, num_signa_ts);
      }
      free(signa_ts);

      ts->qm_timestep->set_gradients(qm_timestep.gradient, numatoms);

      delete [] qm_timestep.scfenergies;
      if (qm_timestep.gradient)     delete [] qm_timestep.gradient;
      if (qm_timestep.charges)      delete [] qm_timestep.charges;
      if (qm_timestep.charge_types) delete [] qm_timestep.charge_types;

      for (i=0; i<qmmeta.num_wavef; i++) {
        delete [] qm_timestep.wave[i].wave_coeffs;
        delete [] qm_timestep.wave[i].orbital_energies;
        delete [] qm_timestep.wave[i].occupancies;
      }
      delete [] qm_timestep.wave;
#if 0
      // If we have at least two timesteps then sort the orbitals
      // of the current timestep according to the ones from the
      // previous timestep.
      // Note that the frame counter m->numframes() is updated
      // after this function call. 
      if (m->numframes()>0) {
        // Get the previous Timestep
        Timestep *prevts = m->get_frame(m->numframes()-1);

        msgInfo << "sort frame " << m->numframes() << sendmsg;

        // Sort the orbitals by comparing them to the ones from
        // the previous timestep.
        ts->qm_timestep->sort_orbitals(prevts->qm_timestep);
      } else {
        msgInfo << "ignore frame " << m->numframes() << sendmsg;
      }
#endif 
    }
  }
  return ts;
}

int MolFilePlugin::skip(Molecule *m) {
  if (!rv) return MOLFILE_ERROR;
  if (!can_read_timesteps()) return MOLFILE_ERROR;
  return plugin->read_next_timestep(rv, numatoms, 0);
}

void MolFilePlugin::close() {
  if (rv && (can_read_structure() || can_read_timesteps() || 
             can_read_graphics() || can_read_volumetric() ||
             can_read_metadata() || can_read_bonds())) { 
    plugin->close_file_read(rv);
    rv = NULL;
  }
  if (wv && (can_write_structure() || can_write_timesteps() ||
             can_write_bonds())) { 
    plugin->close_file_write(wv);
    wv = NULL;
  }
}

int MolFilePlugin::init_write(const char *file, int natoms) {
  wv = NULL;
  if (can_write_structure() || can_write_timesteps() || 
      can_write_volumetric()) { 
    wv = plugin->open_file_write(file, plugin->name, natoms);
  } 
  if (!wv) return MOLFILE_ERROR;
  delete [] _filename;
  _filename = stringdup(file);

  // Cache the number of atoms to be written.  It's not strictly necessary,
  // but it lets us allocate only the necessary space in write_structure
  // and write_timestep.
  numatoms = natoms;

  return MOLFILE_SUCCESS;
}


int MolFilePlugin::write_structure(Molecule *m, const int *on) {
  if (!wv) return MOLFILE_ERROR;
  if (!can_write_structure()) return MOLFILE_ERROR;
  if (!m->has_structure()) {
    msgErr << "Molecule's structure has not been initialized." << sendmsg;
    return MOLFILE_ERROR;
  }
  
  long i, j, k;
  molfile_atom_t *atoms = (molfile_atom_t *) calloc(1, numatoms*sizeof(molfile_atom_t));
  int *atomindexmap = (int *) calloc(1, m->nAtoms*sizeof(int));

  // initialize the atom index map to an invalid atom index value, so that
  // we can use this to eliminate bonds to atoms that aren't selected.
  for (i=0; i<m->nAtoms; i++) 
    atomindexmap[i] = -1;

  const float *charge = m->charge();
  const float *mass = m->mass();
  const float *radius = m->radius();
  const float *beta = m->beta();
  const float *occupancy = m->occupancy();

  int mangleatomnames = 1;
  if (getenv("VMDNOMANGLEATOMNAMES")) {
    mangleatomnames = 0;
  }

  // build the array of selected atoms to be written out
  for (i=0, j=0; i<m->nAtoms; i++) {
    // skip atoms that aren't 'on', if we've been given an 'on' array
    if (on && !on[i]) 
      continue;

    // Check that the number of atoms specified in init_write is no smaller
    // than the number of atoms in the selection; otherwise we would 
    // corrupt memory.
    if (j >= numatoms) {
      msgErr << 
        "MolFilePlugin: Internal error, selection size exceeds numatoms ("
        << numatoms << ")" << sendmsg;
      free(atoms);
      free(atomindexmap);
      return MOLFILE_ERROR;
    }

    const MolAtom *atom = m->atom(i);
    molfile_atom_t &atm = atoms[j];

    if (mangleatomnames) {
      // Try to restore the spacing on the name since VMD destroys it when it
      // reads it in.
      // XXX this is PDB-centric thinking, need to reconsider this
      char name[6], *nameptr;
      name[0] = ' ';
      strncpy(name+1, (m->atomNames).name(atom->nameindex), 4);
      name[5] = '\0';
      // the name must be left-justified
      if(strlen(name) == 5) {
        nameptr = name + 1;
      } else {
        nameptr = name;
        int p;
        while((p = strlen(name)) < 4) {
          name[p] = ' ';
          name[p+1] = '\0';
        }
      }
      strcpy(atm.name, nameptr);
    } else {
      strncpy(atm.name, m->atomNames.name(atom->nameindex), sizeof(atm.name));
    }

    strcpy(atm.type, m->atomTypes.name(atom->typeindex));
    strcpy(atm.resname, m->resNames.name(atom->resnameindex));
    atm.resid = atom->resid;
    strcpy(atm.chain, m->chainNames.name(atom->chainindex));
    strcpy(atm.segid, m->segNames.name(atom->segnameindex));
    strcpy(atm.insertion, atom->insertionstr);
    strcpy(atm.altloc, m->altlocNames.name(atom->altlocindex));
    atm.atomicnumber = atom->atomicnumber;
    atm.occupancy = occupancy[i];
    atm.bfactor = beta[i];
    atm.mass = mass[i]; 
    atm.charge = charge[i]; 
    atm.radius = radius[i];

    atomindexmap[i] = j; // build index map for bond/angle/dihedral/cterms

    j++;
  }

  // check that the selection size matches numatoms
  if (j != numatoms) {
    msgErr 
      << "MolFilePlugin: Internal error, selection size (" << j 
      << ") doesn't match numatoms (" << numatoms << ")" << sendmsg;
    free (atoms);
    return MOLFILE_ERROR;
  }

  // set optional data fields we're providing
  int optflags = MOLFILE_NOOPTIONS; // initialize optflags

  if (m->test_dataset_flag(BaseMolecule::INSERTION))
    optflags |= MOLFILE_INSERTION;

  if (m->test_dataset_flag(BaseMolecule::OCCUPANCY))
    optflags |= MOLFILE_OCCUPANCY;

  if (m->test_dataset_flag(BaseMolecule::BFACTOR))
    optflags |= MOLFILE_BFACTOR;

  if (m->test_dataset_flag(BaseMolecule::MASS))
    optflags |= MOLFILE_MASS;

  if (m->test_dataset_flag(BaseMolecule::CHARGE))
    optflags |= MOLFILE_CHARGE;

  if (m->test_dataset_flag(BaseMolecule::RADIUS))
    optflags |= MOLFILE_RADIUS;
  
  if (m->test_dataset_flag(BaseMolecule::ALTLOC))
    optflags |= MOLFILE_ALTLOC;

  if (m->test_dataset_flag(BaseMolecule::ATOMICNUMBER))
    optflags |= MOLFILE_ATOMICNUMBER;

  // Build and save a bond list if this plugin has a write_bonds() callback.
  // Bonds must be specified to the plugin before write_structure() is called.
  // Only store bond information if it was either set by the user or 
  // by loading from other files.  We don't save auto-generated bond info 
  // by default anymore.  We consider auto-generated bonds worth saving
  // only if the user has customized other bond properties.
  if (can_write_bonds() && 
      (m->test_dataset_flag(BaseMolecule::BONDS) ||
       m->test_dataset_flag(BaseMolecule::BONDORDERS) ||
       m->test_dataset_flag(BaseMolecule::BONDTYPES))) {
    ResizeArray<int> bondfrom, bondto; 
    ResizeArray<float> bondorder;
    ResizeArray<int> bondtype;
    ResizeArray<char *>bondtypename;
    
    float *bondorderptr=NULL;
    int *bondtypeptr=NULL;
    char **bondtypenameptr=NULL;
    int numbondtypes=0;
    
    for (i=0; i<m->nAtoms; i++) {
      const MolAtom *atom = m->atom(i);
      long bfmap = atomindexmap[i] + 1; // 1-based mapped atom index

      for (k=0; k<atom->bonds; k++) {
        int bto = atom->bondTo[k];
        int btmap = atomindexmap[bto] + 1; // 1-based mapped atom index
 
        // add 1-based bonds to 'on' atoms to the bond list
        if (bfmap > 0 && btmap > bfmap) {
          bondfrom.append(bfmap);
          bondto.append(btmap);
          bondorder.append(m->getbondorder(i, k)); 
          bondtype.append(m->getbondtype(i, k));
        }
      }
    } 

    // only store bond orders if they were set by the user or read in 
    // from files 
    if (m->test_dataset_flag(BaseMolecule::BONDORDERS))
      bondorderptr=&bondorder[0];
    else 
      bondorderptr=NULL; // no bond orders provided

    numbondtypes = m->bondTypeNames.num();
    for (i=0; i < numbondtypes; i++) {
      bondtypename.append((char *)m->bondTypeNames.name(i));
    }
    if (numbondtypes > 0)
      bondtypenameptr = &bondtypename[0];
    else 
      bondtypenameptr = NULL;
    
    // only store bond types if they were set by the user or read in 
    // from files 
    if (m->test_dataset_flag(BaseMolecule::BONDTYPES))
      bondtypeptr= &bondtype[0];
    else 
      bondtypeptr=NULL; // no bond types provided

#if vmdplugin_ABIVERSION >= 15
    if (plugin->write_bonds(wv, bondfrom.num(), &bondfrom[0], &bondto[0], 
                            bondorderptr, bondtypeptr, 
                            numbondtypes, bondtypenameptr)) {
#else
    if (plugin->write_bonds(wv, bondfrom.num(), &bondfrom[0], &bondto[0], 
                            bondorderptr)) {
#endif
      free(atoms);
      free(atomindexmap);
      return MOLFILE_ERROR;
    }
  }

  // Only write angle info if all atoms are selected.
  // It's not clear whether there's a point in trying to preserve this
  // kind of information when writing out a sub-structure from an 
  // atom selection.
  if (can_write_angles() && m->test_dataset_flag(BaseMolecule::ANGLES)) {
    ResizeArray<int> angles;
    ResizeArray<int> angleTypes;
    ResizeArray<const char *>angleTypeNames;
    ResizeArray<int> dihedrals;
    ResizeArray<int> dihedralTypes;
    ResizeArray<const char *>dihedralTypeNames;
    ResizeArray<int> impropers;
    ResizeArray<int> improperTypes;
    ResizeArray<const char *>improperTypeNames;
    ResizeArray<int> cterms;

    int numangles = m->num_angles();
    int numdihedrals = m->num_dihedrals();
    int numimpropers = m->num_impropers();

    // generate packed arrays with 1-based indexing
    for (i=0; i<numangles; i++) {
      long i3addr = i*3L;
      int idx0 = atomindexmap[m->angles[i3addr    ]];
      int idx1 = atomindexmap[m->angles[i3addr + 1]];
      int idx2 = atomindexmap[m->angles[i3addr + 2]];
      if ((idx0 >= 0) && (idx1 >= 0) && (idx2 >= 0)) {
        angles.append3(idx0+1, idx1+1, idx2+1); // 1-based indices
#if vmdplugin_ABIVERSION >= 16
        if (m->test_dataset_flag(BaseMolecule::ANGLETYPES)) {
          angleTypes.append(m->get_angletype(i));
        }
#endif
      } 
    } 
    for (i=0; i<numdihedrals; i++) {
      long i4addr = i*4L;
      int idx0 = atomindexmap[m->dihedrals[i4addr    ]];
      int idx1 = atomindexmap[m->dihedrals[i4addr + 1]];
      int idx2 = atomindexmap[m->dihedrals[i4addr + 2]];
      int idx3 = atomindexmap[m->dihedrals[i4addr + 3]];
      if ((idx0 >= 0) && (idx1 >= 0) && (idx2 >= 0) && (idx3 >= 0)) {
        dihedrals.append4(idx0+1, idx1+1, idx2+1, idx3+1); // 1-based indices
#if vmdplugin_ABIVERSION >= 16
        if (m->test_dataset_flag(BaseMolecule::ANGLETYPES)) {
          dihedralTypes.append(m->get_dihedraltype(i));
        }
#endif
      } 
    } 
    for (i=0; i<numimpropers; i++) {
      long i4addr = i*4L;
      int idx0 = atomindexmap[m->impropers[i4addr    ]];
      int idx1 = atomindexmap[m->impropers[i4addr + 1]];
      int idx2 = atomindexmap[m->impropers[i4addr + 2]];
      int idx3 = atomindexmap[m->impropers[i4addr + 3]];
      if ((idx0 >= 0) && (idx1 >= 0) && (idx2 >= 0) && (idx3 >= 0)) {
        impropers.append4(idx0+1, idx1+1, idx2+1, idx3+1); // 1-based indices
#if vmdplugin_ABIVERSION >= 16
        if (m->test_dataset_flag(BaseMolecule::ANGLETYPES)) {
          improperTypes.append(m->get_impropertype(i));
        }
#endif
      } 
    } 

    int ctermcnt=0;
    int *ctermlist=NULL;
    if (m->test_dataset_flag(BaseMolecule::CTERMS)) {
      int numcterms = m->num_cterms();
      for (i=0; i<numcterms; i++) {
        int goodcount=0;
        for (j=0; j<8; j++) {
          if (atomindexmap[m->cterms[i*8L + j]] >= 0)
            goodcount++; 
        }
        if (goodcount == 8) {
          ctermcnt++;
          for (j=0; j<8; j++) {
            // 1-based atom index map
            cterms.append(atomindexmap[m->cterms[i*8L + j]]+1);
          }
        }
      }
      if (ctermcnt > 0)
        ctermlist = &cterms[0];
    }

#if vmdplugin_ABIVERSION >= 16
    // copy all names.
    if (m->test_dataset_flag(BaseMolecule::ANGLETYPES)) {
      for (i=0; i < m->angleTypeNames.num(); i++)
        angleTypeNames.append(m->angleTypeNames.name(i));
      for (i=0; i < m->dihedralTypeNames.num(); i++)
        dihedralTypeNames.append(m->dihedralTypeNames.name(i));
      for (i=0; i < m->improperTypeNames.num(); i++)
        improperTypeNames.append(m->improperTypeNames.name(i));
    }
#endif
    
    int anglescnt         = angles.num()/3;
    int *anglelist        = (anglescnt    > 0) ? &angles[0] : NULL;
#if vmdplugin_ABIVERSION >= 16
    int *angletypelist    = (angleTypes.num() > 0) ? &angleTypes[0] : NULL;
    int angletypecnt      = angleTypeNames.num();
    const char **anglenmlist = (angletypecnt>0) ? &angleTypeNames[0] : NULL;
#endif

    int dihedralscnt      = dihedrals.num()/4;
    int *dihedrallist     = (dihedralscnt > 0) ? &dihedrals[0] : NULL;
#if vmdplugin_ABIVERSION >= 16
    int *dihedraltypelist = (dihedralTypes.num() > 0) ? &dihedralTypes[0] : NULL;
    int dihedraltypecnt   = dihedralTypeNames.num();
    const char **dihedralnmlist = (dihedraltypecnt>0) ? &dihedralTypeNames[0] : NULL;
#endif

    int improperscnt      = impropers.num()/4;
    int *improperlist     = (improperscnt > 0) ? &impropers[0] : NULL;
#if vmdplugin_ABIVERSION >= 16
    int *impropertypelist = (improperTypes.num() > 0) ? &improperTypes[0] : NULL;
    int impropertypecnt   = improperTypeNames.num();
    const char **impropernmlist = (impropertypecnt>0) ? &improperTypeNames[0] : NULL;
    if (plugin->write_angles(wv, anglescnt, anglelist, angletypelist, angletypecnt, 
                             anglenmlist, dihedralscnt, dihedrallist, dihedraltypelist,
                             dihedraltypecnt, dihedralnmlist, improperscnt, 
                             improperlist, impropertypelist, impropertypecnt, 
                             impropernmlist, ctermcnt, ctermlist, 0, 0)) {
      free(atoms);
      free(atomindexmap);
      return MOLFILE_ERROR;
    }
#else
    if (plugin->write_angles(wv, anglescnt, anglelist, NULL,
                             dihedralscnt, dihedrallist, NULL,
                             improperscnt, improperlist, NULL,
                             ctermcnt, ctermlist, 0, 0, NULL)) {
      free(atoms);
      free(atomindexmap);
      return MOLFILE_ERROR;
    }
#endif
  }

  // write the structure
  if (plugin->write_structure(wv, optflags, atoms)) {
    free(atoms);
    free(atomindexmap);
    return MOLFILE_ERROR;
  }

  free(atoms);
  free(atomindexmap);

  return MOLFILE_SUCCESS;
}
 
int MolFilePlugin::write_timestep(const Timestep *ts, const int *on) {
  // it isn't an error if this file format doesn't write timesteps
  if (!can_write_timesteps()) 
    return MOLFILE_SUCCESS; 

  molfile_timestep_t mol_ts;
  memset(&mol_ts, 0, sizeof(molfile_timestep_t));
  mol_ts.A = ts->a_length;
  mol_ts.B = ts->b_length;
  mol_ts.C = ts->c_length;
  mol_ts.alpha = ts->alpha;
  mol_ts.beta = ts->beta;
  mol_ts.gamma = ts->gamma;
  mol_ts.physical_time = ts->physical_time;

  if (!on) {
    mol_ts.coords = ts->pos;
    mol_ts.velocities = ts->vel;
    return plugin->write_timestep(wv, &mol_ts);
  }

  float *coords = new float[3L*numatoms];
  float *vel = NULL;
  if (ts->vel) vel = new float[3L*numatoms];
  long j=0;
  for (int i=0; i<ts->num; i++) {
    if (on[i]) {
      if (on && !on[i]) continue;
      // check that the selection doesn't contain too many atoms
      if (j >= 3L*numatoms) {
        msgErr << "MolFilePlugin::write_timestep: Internal error" << sendmsg;
        msgErr << "Selection size exceeds numatoms (" << numatoms << ")" 
               << sendmsg;
        delete [] coords;
        delete vel;
        return MOLFILE_ERROR;
      }
      coords[j  ] = ts->pos[3L*i];
      coords[j+1] = ts->pos[3L*i+1];
      coords[j+2] = ts->pos[3L*i+2];
      if (ts->vel) {
        vel[j  ] = ts->vel[3L*i  ];
        vel[j+1] = ts->vel[3L*i+1];
        vel[j+2] = ts->vel[3L*i+2];
      }
      j += 3;
    }
  }
  // check that the selection size matches numatoms
  if (j != 3L*numatoms) {
    msgErr << "MolFilePlugin::write_timestep: Internal error" << sendmsg;
    msgErr << "selection size (" << j << ") doesn't match numatoms (" 
           << numatoms << ")" << sendmsg;
    delete [] coords;
    return MOLFILE_ERROR;
  }
  mol_ts.coords = coords;
  mol_ts.velocities = vel;
  int rc = plugin->write_timestep(wv, &mol_ts);
  delete [] coords;
  delete vel;
  return rc;
}
 
int MolFilePlugin::read_rawgraphics(Molecule *m, Scene *sc) {
  if (!rv || !can_read_graphics()) return MOLFILE_ERROR;
  const molfile_graphics_t *graphics = NULL;
  int nelem = -1;
  if (plugin->read_rawgraphics(rv, &nelem, &graphics)) return MOLFILE_ERROR;
  msgInfo << "Reading " << nelem << " graphics elements..." << sendmsg;
  MoleculeGraphics *mg = m->moleculeGraphics();
  for (int i=0; i<nelem; i++) {
    const float *data = graphics[i].data;
    switch (graphics[i].type) {
      case MOLFILE_POINT: 
        mg->add_point(data); 
        break;
      case MOLFILE_TRIANGLE:
        mg->add_triangle(data, data+3, data+6);
        break;
      case MOLFILE_TRINORM:
        {
          const float *ndata;
          // next element must be the norms
          if (i+1 >= nelem || graphics[i+1].type != MOLFILE_NORMS) {
            msgErr << "Invalid rawgraphics: NORMS must follow TRINORM."
              << sendmsg;
            return MOLFILE_ERROR;
          }
          ++i;
          ndata = graphics[i].data;
          mg->add_trinorm(data, data+3, data+6, ndata, ndata+3, ndata+6);
        }
        break;
      case MOLFILE_TRICOLOR: 
        {
          const float *ndata, *cdata;
          // next element must be the norms
          if (i+1 >= nelem || graphics[i+1].type != MOLFILE_NORMS) {
            msgErr << "Invalid rawgraphics: NORMS must follow TRINORM."
              << sendmsg;
            return MOLFILE_ERROR;
          }
          ++i;
          ndata = graphics[i].data;
          // next element must be the vertex colors
          if (i+1 >= nelem || graphics[i+1].type != MOLFILE_COLOR) {
            msgErr << "Invalid rawgraphics: NORMS and COLOR must fullow TRICOLOR."
              << sendmsg;
            return MOLFILE_ERROR;
          }
          ++i;
          cdata = graphics[i].data;
          mg->add_tricolor(data, data+3, data+6, ndata, ndata+3, ndata+6,
              sc->nearest_index(cdata[0], cdata[1], cdata[2]), 
              sc->nearest_index(cdata[3], cdata[4], cdata[5]), 
              sc->nearest_index(cdata[6], cdata[7], cdata[8]));
        }
        break;
      case MOLFILE_LINE:
        mg->add_line(data, data+3, graphics[i].style, (int)graphics[i].size);
        break;
      case MOLFILE_CYLINDER:
        mg->add_cylinder(data, data+3, graphics[i].size, graphics[i].style, 0);
        break;
      case MOLFILE_CAPCYL:
        mg->add_cylinder(data, data+3, graphics[i].size, graphics[i].style, 1);
        break;
      case MOLFILE_CONE:
        mg->add_cone(data, data+3, graphics[i].size, data[6], graphics[i].style);
        break;
      case MOLFILE_SPHERE:
        mg->add_sphere(data, graphics[i].size, graphics[i].style);
        break;
      case MOLFILE_TEXT:
        {
          char text[24];
          strncpy(text, (char *)data+3, 24);
          text[23] = '\0';
          mg->add_text(data, text, graphics[i].size, 1.0f);
        }
        break;
      case MOLFILE_COLOR:
        mg->use_color(sc->nearest_index(data[0], data[1], data[2]));
        break;
      case MOLFILE_NORMS:
        msgErr << "Invalid rawgraphics: NORMS must follow TRINORM." << sendmsg;
        return MOLFILE_ERROR;
        break;
      default:
        msgErr << "Invalid rawgraphics: unknown type " << graphics[i].type
               << sendmsg;
    }
  }

  return MOLFILE_SUCCESS;
}


int MolFilePlugin::read_volumetric(Molecule *m, int nsets, const int *setids) {
  molfile_volumetric_t *metadata; // fetch metadata from file

  int setsinfile = 0;
  plugin->read_volumetric_metadata(rv, &setsinfile, &metadata);

  // Get datasets specified in setids
  int n;
  int *sets;
  if (nsets < 0) {
    n = setsinfile;
    sets = new int[n];
    for (int i=0; i<n; i++) sets[i] = i;
  } else {
    n = nsets;
    sets = new int [n];
    for (int i=0; i<n; i++) sets[i] = setids[i];
  }

  for (int i=0; i< n; i++) {
    if (sets[i] < 0 || sets[i] >= setsinfile) {
      msgErr << "Bogus setid passed to read_volumetric: " << sets[i]
             << sendmsg;
      continue;
    }  

    const molfile_volumetric_t *v = metadata+sets[i];
    size_t size = long(v->xsize) * long(v->ysize) * long(v->zsize);

    char *dataname = stringdup(v->dataname);
    if (_filename) {
      // prepend the filename to the dataname; otherwise it's hard to tell
      // multiple data sets apart in the GUI.  This should be done here,
      // within VMD, rather than within each plugin because otherwise 
      // different plugins will end up exhibiting different behavior with
      // regard to naming their datasets.
      //
      // XXX The breakup_filename command uses forward slashes only and
      // is therefore Unix-specific.  Also, to avoid super-long dataset
      // names I'm going to use just the 'basename' part of the file.
      // It's easier just to code a correct version of what I want here.
      char sep = 
#ifdef WIN32
        '\\'
#else
        '/'
#endif
        ;
      const char *basename = strrchr(_filename, sep);
      if (!basename) {
        basename = _filename;
      } else {
        basename++; // skip the separator
      }
      char *tmp = new char[strlen(dataname)+5+strlen(basename)];
      sprintf(tmp, "%s : %s", basename, dataname);
      delete [] dataname;
      dataname = tmp;
    }


#if vmdplugin_ABIVERSION > 16
    if (plugin->read_volumetric_data_ex != NULL) {
msgInfo << "Loading voumetric data using ABI 17+ ... " << sendmsg;
      molfile_volumetric_readwrite_t rwparms;
      memset(&rwparms, 0, sizeof(rwparms));
 
      rwparms.setidx = sets[i];
      if (v->has_scalar) 
        rwparms.scalar = new float[size];
      if (v->has_gradient) 
        rwparms.gradient = new float[3L*size];
#if 0
      if (v->has_variance) 
        rwparms.variance = new float[size];
      if (v->has_color == ...) 
        rwparms.rgb3f = new float[3L*size];
      if (v->has_color == ...) 
        rwparms.rgb3u = new unsigned char[3L*size];
#endif

      if (plugin->read_volumetric_data_ex(rv, &rwparms)) {
        msgErr << "Error reading volumetric data set " << sets[i]+1 << sendmsg;
        delete [] dataname;
        delete [] rwparms.scalar;
        delete [] rwparms.gradient;
        delete [] rwparms.variance;
        delete [] rwparms.rgb3f;
        delete [] rwparms.rgb3u;
        continue;
      }

      m->add_volume_data(dataname, v->origin, v->xaxis, v->yaxis, v->zaxis, 
                         v->xsize, v->ysize, v->zsize, 
                         rwparms.scalar, rwparms.gradient, rwparms.variance);
      delete [] dataname;
    } else if (plugin->read_volumetric_data != NULL) {
#endif 
      float *scalar=NULL, *rgb3f=NULL;
      scalar = new float[size];
      if (v->has_color) 
        rgb3f = new float[3L*size];
      if (plugin->read_volumetric_data(rv, sets[i], scalar, rgb3f)) {
        msgErr << "Error reading volumetric data set " << sets[i]+1 << sendmsg;
        delete [] dataname;
        delete [] scalar;
        delete [] rgb3f;
        continue;
      }

      m->add_volume_data(dataname, v->origin, v->xaxis, v->yaxis, v->zaxis, 
                         v->xsize, v->ysize, v->zsize, scalar);
      delete [] dataname;
      delete [] rgb3f; // have to delete if created, since we don't use yet
#if vmdplugin_ABIVERSION > 16
    }
#endif 
  }

  delete [] sets;

  return MOLFILE_SUCCESS;
}


int MolFilePlugin::read_metadata(Molecule *m) {
  // Fetch metadata from file
  molfile_metadata_t *metadata;

  plugin->read_molecule_metadata(rv, &metadata);

  m->record_database(metadata->database, metadata->accession);
  if (metadata->remarks != NULL) 
    m->record_remarks(metadata->remarks);
  else 
    m->record_remarks("");

  return MOLFILE_SUCCESS;
}


int MolFilePlugin::read_qm_data(Molecule *mol) {
  // Fetch metadata from file.
  // It provides us with a bunch of sizes for the arrays
  // that we have to allocate and provide to read_qm_rundata().
  // Note that this probably should be done 
  molfile_qm_metadata_t metadata;

  // check for failures while parsing metadata and bail out if
  // an error occurs.
  if (plugin->read_qm_metadata(rv, &metadata) != MOLFILE_SUCCESS)
    return MOLFILE_ERROR;

  // If the plugin didn't provide the number of atoms
  // (e.g. because it only read the basis set) we set it
  // to the number of atoms in the molecule.
  // XXX This is kind of dangerous: If you load a basis set
  // on top of a multimillion atom structure then a basis
  // will be added to each of the atoms whic cost a lot of
  // memory!
  if (!numatoms) numatoms = mol->nAtoms;
  mol->qm_data = new QMData(numatoms,
                            metadata.num_basis_funcs,
                            metadata.num_shells,
                            metadata.wavef_size);
  //mol->qm_data->nintcoords = metadata.nintcoords;

  // We need to keep a pointer to the QMData object
  // to be used in next() when we are sorting the
  // wavefunction coefficients for the current timestep.
  qm_data = mol->qm_data;

  molfile_qm_t *qmdata = (molfile_qm_t *) calloc(1, sizeof(molfile_qm_t));

  // Allocate memory for the arrays:
  if (metadata.num_basis_atoms) {
    qmdata->basis.num_shells_per_atom = new int[metadata.num_basis_atoms];
    qmdata->basis.atomic_number       = new int[metadata.num_basis_atoms];
    qmdata->basis.num_prim_per_shell  = new int[metadata.num_shells];
    qmdata->basis.basis            = new float[2L*metadata.num_basis_funcs];
    qmdata->basis.shell_types      = new int[metadata.num_shells];
    qmdata->basis.angular_momentum = new int[3L*metadata.wavef_size];
  }

  if (metadata.nimag)
    qmdata->hess.imag_modes = new int[metadata.nimag];

  if (metadata.have_normalmodes) {
    qmdata->hess.normalmodes = new float[metadata.ncart*metadata.ncart];
    qmdata->hess.wavenumbers = new float[metadata.ncart];
    qmdata->hess.intensities = new float[metadata.ncart];
  }

  if (metadata.have_carthessian)
    qmdata->hess.carthessian = new double[metadata.ncart*metadata.ncart];

  if (metadata.have_inthessian)
    qmdata->hess.inthessian = new double[metadata.nintcoords*metadata.nintcoords];

//   if (metadata.have_esp) {
//     qmdata->run.esp_charges = new double[numatoms];
//   }
//   else
//     qmdata->run.esp_charges = NULL;

  // All necessary arrays are allocated.
  // Now get the data from the plugin:
  plugin->read_qm_rundata(rv, qmdata);

  // Copy data from molfile_plugin structs into VMD's data structures.

  if (metadata.have_sysinfo) {
    //mol->qm_data->num_orbitals_A = qmdata->run.num_orbitals_A;
    //mol->qm_data->num_orbitals_B = qmdata->run.num_orbitals_B;
    mol->qm_data->nproc  = qmdata->run.nproc;
    mol->qm_data->memory = qmdata->run.memory;

    // We need to translate between the macros used in the plugins
    // and the one ones in VMD, here so that they can be independent
    // and we don't have to include molfile_plugin.h anywhere else
    // in VMD.
    switch (qmdata->run.scftype) {
    case MOLFILE_SCFTYPE_NONE:
      mol->qm_data->scftype = SCFTYPE_NONE;
      break;
    case MOLFILE_SCFTYPE_RHF:
      mol->qm_data->scftype = SCFTYPE_RHF;
      break;
    case MOLFILE_SCFTYPE_UHF:
      mol->qm_data->scftype = SCFTYPE_UHF;
      break;
    case MOLFILE_SCFTYPE_ROHF:
      mol->qm_data->scftype = SCFTYPE_ROHF;
      break;
    case MOLFILE_SCFTYPE_GVB:
      mol->qm_data->scftype = SCFTYPE_GVB;
      break;
    case MOLFILE_SCFTYPE_MCSCF:
      mol->qm_data->scftype = SCFTYPE_MCSCF;
      break;
    case MOLFILE_SCFTYPE_FF:
      mol->qm_data->scftype = SCFTYPE_FF;
      break;
    default:
      mol->qm_data->scftype = SCFTYPE_UNKNOWN;
    }

    switch (qmdata->run.runtype) {
    case MOLFILE_RUNTYPE_ENERGY:
      mol->qm_data->runtype = RUNTYPE_ENERGY;
      break;
    case MOLFILE_RUNTYPE_OPTIMIZE:
      mol->qm_data->runtype = RUNTYPE_OPTIMIZE;
      break;
    case MOLFILE_RUNTYPE_SADPOINT:
      mol->qm_data->runtype = RUNTYPE_SADPOINT;
      break;
    case MOLFILE_RUNTYPE_HESSIAN:
      mol->qm_data->runtype = RUNTYPE_HESSIAN;
      break;
    case MOLFILE_RUNTYPE_SURFACE:
      mol->qm_data->runtype = RUNTYPE_SURFACE;
      break;
    case MOLFILE_RUNTYPE_GRADIENT:
      mol->qm_data->runtype = RUNTYPE_GRADIENT;
      break;
    case MOLFILE_RUNTYPE_MEX:
      mol->qm_data->runtype = RUNTYPE_MEX;
      break;
    case MOLFILE_RUNTYPE_DYNAMICS:
      mol->qm_data->runtype = RUNTYPE_DYNAMICS;
      break;
    case MOLFILE_RUNTYPE_PROPERTIES:
      mol->qm_data->runtype = RUNTYPE_PROPERTIES;
      break;
    default:
      mol->qm_data->runtype = RUNTYPE_UNKNOWN;
    }

    switch (qmdata->run.status) {
    case MOLFILE_QMSTATUS_OPT_CONV:
      mol->qm_data->status = QMSTATUS_OPT_CONV;
      break;
    case MOLFILE_QMSTATUS_OPT_NOT_CONV:
      mol->qm_data->status = QMSTATUS_OPT_NOT_CONV;
      break;
    case MOLFILE_QMSTATUS_SCF_NOT_CONV:
      mol->qm_data->status = QMSTATUS_SCF_NOT_CONV;
      break;
    case MOLFILE_QMSTATUS_FILE_TRUNCATED:
      mol->qm_data->status = QMSTATUS_FILE_TRUNCATED;
      break;
    default:
      mol->qm_data->status = QMSTATUS_UNKNOWN;
    }

    // Run data:
    SAFESTRNCPY(mol->qm_data->version_string, qmdata->run.version_string);
    SAFESTRNCPY(mol->qm_data->runtitle, qmdata->run.runtitle);
    SAFESTRNCPY(mol->qm_data->geometry, qmdata->run.geometry);
  }

  if (metadata.have_sysinfo) {
    // Initialize total charge, multiplicity, number of electrons,
    // number of occupied orbitals.
    // Note that mol->qm_data->scftyp must have been
    // assign before.
    mol->qm_data->init_electrons(mol, qmdata->run.totalcharge);
  }

  // Populate basis set data and organize them into
  // hierarcical data structures.
  if (!mol->qm_data->init_basis(mol, metadata.num_basis_atoms,
                                qmdata->run.basis_string,
                                qmdata->basis.basis,
                                qmdata->basis.atomic_number,
                                qmdata->basis.num_shells_per_atom,
                                qmdata->basis.num_prim_per_shell,
                                qmdata->basis.shell_types)) {
    msgWarn << "Incomplete basis set info in QM data."
           << sendmsg;
  }

  // Exponents of angular momenta in wave function
  if (metadata.wavef_size) {
    mol->qm_data->set_angular_momenta(qmdata->basis.angular_momentum);
  }


  // Hessian data:
  if (metadata.have_carthessian) {
    mol->qm_data->set_carthessian(metadata.ncart, qmdata->hess.carthessian);
  }
  
  if (metadata.have_inthessian) {
    mol->qm_data->set_inthessian(metadata.nintcoords, qmdata->hess.inthessian);
  }

  if (metadata.have_normalmodes) {
    mol->qm_data->set_normalmodes(metadata.ncart, qmdata->hess.normalmodes);
    mol->qm_data->set_wavenumbers(metadata.ncart, qmdata->hess.wavenumbers);
    mol->qm_data->set_intensities(metadata.ncart, qmdata->hess.intensities);
  }

  if (metadata.nimag) {
    mol->qm_data->set_imagmodes(metadata.nimag, qmdata->hess.imag_modes);
  }

  // Cleanup the arrays we needed to get the data from the plugin.
  if (metadata.num_basis_atoms) {
    delete [] qmdata->basis.num_shells_per_atom;
    delete [] qmdata->basis.atomic_number;
    delete [] qmdata->basis.num_prim_per_shell;
    delete [] qmdata->basis.basis;
    delete [] qmdata->basis.shell_types;
    delete [] qmdata->basis.angular_momentum;
  }
  delete [] qmdata->hess.carthessian;
  delete [] qmdata->hess.inthessian;
  delete [] qmdata->hess.normalmodes;
  delete [] qmdata->hess.wavenumbers;
  delete [] qmdata->hess.intensities;
  delete [] qmdata->hess.imag_modes;
  free(qmdata);

  return MOLFILE_SUCCESS;
}


int MolFilePlugin::write_volumetric(Molecule *m, int set) {
  if (set < 0 || set > m->num_volume_data()) {
    msgErr << "Bogus setid passed to write_volumetric: " << set
           << sendmsg;
    return MOLFILE_SUCCESS;
  } 

  const VolumetricData *v = m->get_volume_data(set); 

  molfile_volumetric_t volmeta;

  // SAFESTRNCPY(volmeta.dataname, v->name); 
  int n = ((sizeof(volmeta.dataname) < strlen(v->name)+1) ? 
            sizeof(volmeta.dataname) : strlen(v->name)+1);
  strncpy(volmeta.dataname, v->name, n);

  for (int i=0; i<3; i++) {
      volmeta.origin[i] = (float)v->origin[i];
      volmeta.xaxis[i] = (float)v->xaxis[i];
      volmeta.yaxis[i] = (float)v->yaxis[i];
      volmeta.zaxis[i] = (float)v->zaxis[i];
  }
  volmeta.xsize = v->xsize;
  volmeta.ysize = v->ysize;
  volmeta.zsize = v->zsize;
  volmeta.has_color = 0;
 
  float *datablock = v->data;
  float *colorblock = NULL;

  plugin->write_volumetric_data(wv, &volmeta, datablock, colorblock);

  return MOLFILE_SUCCESS;
}



