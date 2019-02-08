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
 *      $RCSfile: MolFilePlugin.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.45 $      $Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   VMD interface to 'molfile' plugins.  Molfile plugins read coordinate
 *   files, structure files, volumetric data, and graphics data.  The data
 *   is loaded into a new or potentially preexisting molecule in VMD.
 * 
 * LICENSE:
 *   UIUC Open Source License
 *   http://www.ks.uiuc.edu/Research/vmd/plugins/pluginlicense.html
 * 
 ***************************************************************************/
#ifndef MOL_FILE_PLUGIN_H__
#define MOL_FILE_PLUGIN_H__

class Molecule;
class Timestep;
class AtomSel;
class Scene;
class QMData;

#include <stdlib.h>  // for NULL
#include "molfile_plugin.h"

/// VMD interface to 'molfile' plugins.  Molfile plugins read coordinate
/// files, structure files, volumetric data, and graphics data.  The data
/// is loaded into a new or potentially preexisting molecule in VMD.
class MolFilePlugin {
private:
  molfile_plugin_t *plugin;
  void *rv;                  ///< reader file handle from the plugin
  void *wv;                  ///< writer file handle from the plugin
  int numatoms;              ///< number of atoms either to read or to write
  char *_filename;           ///< private copy of the filename
  QMData *qm_data;           ///< pointer to the QMData object in Molecule
  void close();              ///< close file handle and delete _filename

public:
  MolFilePlugin(vmdplugin_t *p);
  virtual ~MolFilePlugin();

  // test whether the plugin supports various capabilities 
  const char *name() const        { return plugin->name; }
  const char *prettyname() const  { return plugin->prettyname; }
  const char *extension() const   { return plugin->filename_extension; }

  int can_read_structure() const  { return plugin->read_structure != NULL; }
  int can_read_bonds() const      { return plugin->read_bonds != NULL; }
  int can_read_pagealigned_timesteps() const { 
#if vmdplugin_ABIVERSION > 17
    return plugin->read_timestep_pagealign_size != NULL; 
#else
    return 0;
#endif
  }
  int can_read_timesteps() const  { return plugin->read_next_timestep != NULL;}
  int can_read_graphics() const   { return plugin->read_rawgraphics != NULL; }
  int can_read_volumetric() const { return plugin->read_volumetric_metadata != NULL; }
  int can_read_metadata() const   { return plugin->read_molecule_metadata != NULL; }
  int can_read_qm() const         { return plugin->read_qm_metadata != NULL; }
  int can_read_qm_timestep()      { return plugin->read_timestep != NULL; }
  int can_read_angles()           { return plugin->read_angles != NULL; }
  int can_read_timestep_metadata() { return plugin->read_timestep_metadata != NULL; }
  int can_read_qm_timestep_metadata() { return plugin->read_qm_timestep_metadata != NULL; }

  int can_write_structure() const { return plugin->write_structure != NULL; }
  int can_write_bonds() const     { return plugin->write_bonds != NULL; }
  int can_write_timesteps() const { return plugin->write_timestep != NULL; }
  int can_write_angles()           { return plugin->read_angles != NULL; }
  int can_write_volumetric() const { return plugin->write_volumetric_data != NULL; }

  int init_read(const char *file);        ///< open file for reading; 
                                          ///< read natoms.  Return 0
                                          ///< on success. 

  int natoms() const { return numatoms; } ///< Number of atoms in each timestep
                                          ///< -1 if unknown (not in file)
  void set_natoms(int);                   ///< override the number of atoms
                                          ///< in each step; this is needed
                                          ///< for CRD files  :-(
 
  int read_structure(Molecule *m, int filebonds, int autobonds); ///< init mol,
                                          ///< file bond determination
                                          ///< auto bond determination
                                          ///< return 0 on success.

  /// Read optional structure information into the molecule, but without 
  /// initializing atoms.  Sets all optional fields like radius and charge,
  /// as well as bonds, but only if the file explicitly contains it.  If
  /// bonds are found then the molecular topology is recalculated as well.
  int read_optional_structure(Molecule *m, int filebonds);

  /// Query the molfile plugin to determine whether or not memory
  /// allocations used for atomic coordinates and PBC unit cell information
  /// need to be aligned to a particular virtual memory or filesystem 
  /// page size boundary to facilitate kernel-bypass unbuffered I/O,
  /// e.g., as used by jsplugin.
  /// This API should be called prior to the first call to read a timestep.
  /// The required page alignment size (in bytes) is returned to the caller.
  /// If this API has not been called, then the molfile plugin should revert 
  /// to standard kernel-buffered I/O and suffer the associated performance 
  /// loss. If a page-aligned allocation is not required for the file 
  /// being read, the plugin will return an alignment size of 1.
  /// On error, a value of -1 will be returned.
  int read_timestep_pagealign_size(void);

  Timestep *next(Molecule *m, int ts_pagealign_sz=1); ///< next timestep
  int skip(Molecule *m);                 ///< skip over a step; return 0 on success.

  /// Read raw graphics data into the given molecule
  int read_rawgraphics(Molecule *, Scene *);

  /// Read selected volumetric datasets into molecule.  Return 0 on success.
  /// If nsets == -1 then all sets will be read.
  int read_volumetric(Molecule *, int nsets, const int *setids);

  /// Read file metadata into molecule.  Return 0 on success.
  int read_metadata(Molecule *);

  /// Read QM data into molecule.  Return 0 on success.
  int read_qm_data(Molecule *);

  int init_write(const char *file, int natoms);

  /// if can_write_structure returns true, then write_structure must be called 
  /// before the first invocation of write_timestep.  If sel is non-NULL,
  /// it must point to an array of indices, one for each atom in Molecule
  /// or Timestep, indicating which atoms are to be written to the file.
  int write_structure(Molecule *, const int *sel);
  int write_timestep(const Timestep *, const int *sel); 

  /// Write volumetric data
  int write_volumetric(Molecule *, int set);

};

#endif

