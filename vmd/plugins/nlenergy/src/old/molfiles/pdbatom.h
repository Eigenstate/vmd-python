/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    molfiles/pdbatom.h
 * @brief   Read and write PDB atoms (ATOM and HETATM records).
 * @author  David J. Hardy
 * @date    Aug. 2007
 */

#ifndef MOLFILES_PDBATOM_H
#define MOLFILES_PDBATOM_H

#include "moltypes/moltypes.h"

#ifdef __cplusplus
extern "C" {
#endif

  /**@brief PDB @c ATOM and @c HETATM record information.
   *
   * Auxiliary structure for information from PDB @c ATOM or @c HETATM
   * records.  The field names and notes below are from the PDB format
   * description guide:
   * http://www.rcsb.org/pdb/docs/format/pdbguide2.2/guide2.2_frame.html .
   * The extra space in the @c char arrays are for nil-termination and
   * 32-bit padding.  The column numbering given is FORTRAN-based,
   * ranging from 1 to 80.
   */
  typedef struct PdbAtomAux_t {
    /*
     *                        definition                         columns
     */
    char record[8];  /**< Record name: "ATOM  " or "HETATM"   (col:  1 -  6) */
    char serial[8];  /**< Atom serial number.                 (col:  7 - 11) */
    char name[8];    /**< Atom name.                          (col: 13 - 16) */
    char altLoc[4];  /**< Alternate location identifier.      (col:      17) */
    char resName[4]; /**< Residue name.                       (col: 18 - 20) */
    char chainID[4]; /**< Chain identifier.                   (col:      22) */
    char resSeq[8];  /**< Residue sequence number.            (col: 23 - 26) */
    char iCode[4];   /**< Code for insertion of residues.     (col:      27) */
    freal occupancy; /**< Occupancy.                          (col: 55 - 60) */
    freal tempFactor;/**< Temperature factor.                 (col: 61 - 66) */
    char segID[8];   /**< Segment identifier, left-justified. (col: 73 - 76) */
    char element[4]; /**< Element symbol, right-justified.    (col: 77 - 78) */
    char charge[4];  /**< Charge on the atom.                 (col: 79 - 80) */
  } PdbAtomAux;

  /**@ Container for ATOM record data from PDB file. */
  typedef struct PdbAtom_t {
    Array atomCoord;   /**< array of dvec (could be pos, vel, or other) */
    Array pdbAtomAux;  /**< array of PdbAtomAux */
  } PdbAtom;

  /**@brief Designate unit conversion of coordinates between file and array. */
  typedef enum PdbAtomType_t {
    PDBATOM_NONE = 0,  /**< no unit conversion */
    PDBATOM_POS  = 0,  /**< Angstrom file, Angstrom array */
    PDBATOM_VEL  = 1   /**< A/ps file, A/fs array */
  } PdbAtomType;

  int PdbAtom_init(PdbAtom *);
  void PdbAtom_done(PdbAtom *);

  /**@brief Either set n to number of expected atoms or leave as zero,
   * meaning that _read() will expand array as needed. */
  int PdbAtom_setup(PdbAtom *, int32 n);

  /* access functions */
  int32              PdbAtom_numatoms   (const PdbAtom *);
  const dvec       * PdbAtom_coord_const(const PdbAtom *);
  const PdbAtomAux * PdbAtom_aux_const  (const PdbAtom *);
  dvec             * PdbAtom_coord      (PdbAtom *);
  PdbAtomAux       * PdbAtom_aux        (PdbAtom *);

  int PdbAtom_set_coord(PdbAtom *, const dvec *, int32 n);
  int PdbAtom_set_aux(PdbAtom *, const PdbAtomAux *, int32 n);

  /* file I/O */
  int PdbAtom_read(PdbAtom *, PdbAtomType atype, const char *fname);
  int PdbAtom_write(const PdbAtom *, PdbAtomType atype, const char *fname);

#ifdef __cplusplus
}
#endif

#endif /* MOLFILES_PDBATOM_H */
