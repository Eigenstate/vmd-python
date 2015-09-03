/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: Molecule.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.61 $	$Date: 2010/12/16 04:08:24 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Main Molecule objects, which contains all the capabilities necessary to
 * store, draw, and manipulate a molecule.  This adds to the functions of
 * DrawMolecule and BaseMolecule by adding file I/O interfaces
 *
 ***************************************************************************/
#ifndef MOLECULE_H
#define MOLECULE_H

#include "DrawMolecule.h"
#include "ResizeArray.h"
#include "utilities.h"

class VMDApp;
class CoorData;

/// Main Molecule objects, which contains all the capabilities necessary to
/// store, draw, and manipulate a molecule.  This adds to the functions of
/// DrawMolecule and BaseMolecule by adding file I/O interfaces etc.
class Molecule : public DrawMolecule {

private:
  ResizeArray<char *> fileList;       ///< files loaded into this molecule
  ResizeArray<char *> fileSpecList;  ///< file loading options used
  ResizeArray<char *> typeList;       ///< file types loaded into this molecule
  ResizeArray<char *> dbList;         ///< database each file came from
  ResizeArray<char *> accessionList;  ///< database access code for each file
  ResizeArray<char *> remarksList;    ///< freeform text comments for each file
    
public:
  /// constructor ... nothing much to do, just save the source, and pass on
  /// the other data.
  /// Pass the VMDApp and parent Displayable on to DrawMolecule
  Molecule(const char *, VMDApp *, Displayable *);
  
  /// destructor ... just clear out unread files
  virtual ~Molecule(void);

  /// rename the molecule, as shown in the GUI and as accessed
  /// by text commands
  int rename(const char *newname);

  /// Return the number of file components that make up the loaded molecule
  int num_files() const { return fileList.num(); }
 
  /// Return the file type of the indexed molecule file component
  const char *get_type(int i) const {
    if (i < 0 || i >= typeList.num()) return NULL;
    return typeList[i];
  } 

  /// Return the file name of the indexed molecule file component
  const char *get_file(int i) const {
    if (i < 0 || i >= fileList.num()) return NULL;
    return fileList[i];
  } 

  /// Return the file loading specs of the indexed molecule file component
  const char *get_file_specs(int i) const {
    if (i < 0 || i >= fileSpecList.num()) return NULL;
    return fileSpecList[i];
  } 

  /// Return the source database of the indexed molecule file component
  const char *get_database(int i) const {
    if (i < 0 || i >= dbList.num()) return NULL;
    return dbList[i];
  }

  /// Return the database accession code of the indexed molecule file component
  const char *get_accession(int i) const {
    if (i < 0 || i >= accessionList.num()) return NULL;
    return accessionList[i];
  }

  /// Return the text comments for the indexed molecule file component
  const char *get_remarks(int i) const {
    if (i < 0 || i >= remarksList.num()) return NULL;
    return remarksList[i];
  }

  /// Record the file component's name, type, and loading specs in the molecule
  void record_file(const char *filename, const char *filetype, const char *filespecs) {
    fileList.append(stringdup(filename));
    typeList.append(stringdup(filetype));
    fileSpecList.append(stringdup(filespecs));
  }

  /// Record the file component's source database and accession code 
  /// in the molecule
  void record_database(const char *dbname, const char *dbcode) {
    dbList.append(stringdup(dbname));
    accessionList.append(stringdup(dbcode));
  }

  /// Record the file component's text comments in the molecule
  void record_remarks(const char *remarks) {
    remarksList.append(stringdup(remarks));
  }
    
  /// add a CoorFile object to the file I/O queue.  Molecule will read/write
  /// frames until the CoorFileData object says its finished.  Then Molecule
  /// will delete it.
  void add_coor_file(CoorData *);

  /// Complete/close a CoorFile object.  Molecule will signal any necessary
  /// callbacks etc.
  void close_coor_file(CoorData *);

  /// Check for new trajectory frames in I/O queue and IMD
  int get_new_frames();

  /// Read the next frame in the file I/O queue.  Return true if any frames
  /// were read; otherwise return false.
  int next_frame();
  
  /// cancel loading/saving of all coordinate files
  /// return number of files canceled
  int cancel();

  /// return true if file I/O is in progress
  int file_in_progress() { return coorIOFiles.num(); }

  /// prepare for drawing ... do any updates needed right before draw.
  /// This possibly reads a new timestep, if requested.
  virtual void prepare();

  // Forces: Various UIObjects add forces to the molecule during the 
  // check_event stage.  In prepare, Molecule sums all the forces, sets
  // the forces in Timestep, and calls app->imd_sendforces() if there's
  // anything to send.  Only Molecule should modify the forces in Timestep
  // directly.

  /// add a force <f> to atom #<theatom> for this display loop only 
  void addForce(int theatom, const float * f);

  /// add a persistent force to the given atom; this will stay until cleared
  /// (by making the force zero).  This force will be added to any other forces
  /// coming from addForce.
  void addPersistentForce(int theatom, const float * f);
  
private:
  /// An array of forces for all the atoms.  These will be built up
  /// incrementally then stuck into the timestep.
  ResizeArray<int> force_indices;
  ResizeArray<float> force_vectors;

  ResizeArray<int> persistent_force_indices;
  ResizeArray<float> persistent_force_vectors;

  /// The atoms that had applied forces in the last timestep
  ResizeArray<int> last_force_indices;

  /// Coordinate files we are supposed to read/write. Only one will be accessed
  /// at a time, though, so we must maintain a list.
  ResizeArray<CoorData *> coorIOFiles;
};

#endif

