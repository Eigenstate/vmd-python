/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 */

#include <string.h>
#include "molfiles/psf.h"

#undef  FEOF
#define FEOF  1


static int psf_reader(NL_FILE *f, Topology *t);


int Topology_read_psf(Topology *t, const char *fname)
{
  NL_FILE *f;
  int s;

  if (NULL==(f = NL_fopen(fname, "r"))) return ERROR(ERR_FOPEN);
  else if ((s=psf_reader(f, t)) != OK) {
    NL_fclose(f);
    return ERROR(s);
  }
  else if (NL_fclose(f)) return ERROR(ERR_FCLOSE);
  return OK;
}


static int psf_get_line(NL_FILE *f, char *line, int len);
static int psf_read_header(NL_FILE *f);
static int psf_read_atoms(NL_FILE *f, Topology *t);
static int psf_read_bonds(NL_FILE *f, Topology *t);
static int psf_read_angles(NL_FILE *f, Topology *t);
static int psf_read_dihedrals(NL_FILE *f, Topology *t);
static int psf_read_impropers(NL_FILE *f, Topology *t);
static int psf_read_donors(NL_FILE *f, Topology *t);
static int psf_read_acceptors(NL_FILE *f, Topology *t);
static int psf_read_exclusions(NL_FILE *f, Topology *t);


int psf_reader(NL_FILE *f, Topology *t)
{
  int s;

  if ((s=psf_read_header(f)) != OK) return ERROR(s);
  if ((s=psf_read_atoms(f,t)) != OK) return ERROR(s);
  if ((s=psf_read_bonds(f,t)) != OK) return ERROR(s);
  if ((s=psf_read_angles(f,t)) != OK) return ERROR(s);
  if ((s=psf_read_dihedrals(f,t)) != OK) return ERROR(s);
  if ((s=psf_read_impropers(f,t)) != OK) return ERROR(s);
  if ((s=psf_read_donors(f,t)) != OK) return ERROR(s);
  if ((s=psf_read_acceptors(f,t)) != OK) return ERROR(s);
  if ((s=psf_read_exclusions(f,t)) != OK) return ERROR(s);

  /* ignore remaining file content after NNB section */

  return OK;
}


int psf_get_line(NL_FILE *f, char *line, int len)
{
  ASSERT(len >= 3);
  /* make sure file line length does not exceed len-2 characters */
  line[len-2] = '\n';
  line[len-3] = '\n';
  if (NL_fgets(line, len, f) == NULL) {
    if (NL_feof(f)) return FEOF;
    else return ERROR(ERR_FREAD);
  }
  else if (line[len-2] != '\n' && line[len-3] != '\n') return ERROR(ERR_INPUT);
  return OK;
}


#define BUFLEN  512
#define TOKLEN  16
#define TOKEN   "%15[a-zA-Z0-9]"
#define ANAME   "%4s"
#define ASKIP   "%*4s"
#define INT32   FMT_INT32
#define ISKIP   "%*d"
#define DREAL   FMT_DREAL
#define WS      " "

#define NELEMS(arr)  (sizeof(arr)/sizeof(arr[0]))


int psf_read_header(NL_FILE *f)
{
  char buf[BUFLEN];
  char tok[TOKLEN];
  int32 n, i;
  int s;

  /* parse PSF keyword */
  if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
    return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
  }
  if (sscanf(buf, WS TOKEN, tok) != 1
      || strcasecmp(tok, "PSF") != 0) {
    return ERROR(ERR_INPUT);  /* expecting PSF keyword */
  }

  /* parse NTITLE section */
  do {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
  } while (sscanf(buf, WS TOKEN, tok) != 1);  /* skip blank lines */
  if (sscanf(buf, INT32 " ! " TOKEN, &n, tok) != 2
      || strcasecmp(tok, "NTITLE") != 0) {
    return ERROR(ERR_INPUT);  /* expecting NTITLE keyword */
  }
  /* read (and discard) REMARKS */
  for (i = 0;  i < n;  i++) {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
    if (sscanf(buf, WS TOKEN, tok) != 1
        || strcasecmp(tok, "REMARKS") != 0) {
      return ERROR(ERR_INPUT);  /* expecting line of REMARKS */
    }
  }
  return OK;
}


int psf_read_atoms(NL_FILE *f, Topology *t)
{
  char buf[BUFLEN];
  char tok[TOKLEN];
  Atom atom;
  int32 n, i, k, s;

  if (Topology_atom_array_length(t) != 0) return ERROR(ERR_VALUE);

  /* parse NATOM section */
  do {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
  } while (sscanf(buf, WS TOKEN, tok) != 1);  /* skip blank lines */
  if (sscanf(buf, INT32 " ! " TOKEN, &n, tok) != 2
      || strcasecmp(tok, "NATOM") != 0) {
    return ERROR(ERR_INPUT);  /* expecting NATOM keyword */
  }

  if ((s=Topology_setmaxnum_atom(t, n)) != OK) return ERROR(s);

  /* read and parse atom records */
  for (i = 0;  i < n;  i++) {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
    if (sscanf(buf, INT32 WS ASKIP WS INT32 WS ANAME WS
          ANAME WS ANAME WS DREAL WS DREAL, &k, &atom.residue, atom.resName,
          atom.atomName, atom.atomType, &atom.q, &atom.m) != 7) {
      return ERROR(ERR_INPUT);  /* unable to parse atom record */
    }
    else if (i+1 != k) {
      return ERROR(ERR_INPUT);  /* atom record is misnumbered */
    }
    atom.residue--;        /* adjust for 0-based indexing */
    atom.atomPrmID = -1;   /* not yet established */
    atom.clusterID = -1;
    atom.clusterSize = 0;
    atom.atomInfo = 0;
    if ((s=Topology_add_atom(t, &atom)) != i) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
  }
  return OK;
}


int psf_read_bonds(NL_FILE *f, Topology *t)
{
  char buf[BUFLEN];
  char tok[TOKLEN];
  Bond bond;
  int32 n, i, j, s;
  const char *bondfmt[] = {
    INT32 WS INT32,
    ISKIP WS ISKIP WS INT32 WS INT32,
    ISKIP WS ISKIP WS ISKIP WS ISKIP WS INT32 WS INT32,
    ISKIP WS ISKIP WS ISKIP WS ISKIP WS ISKIP WS ISKIP WS INT32 WS INT32
  };

  if (Topology_bond_array_length(t) != 0) return ERROR(ERR_VALUE);

  /* parse NBOND section */
  do {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
  } while (sscanf(buf, WS TOKEN, tok) != 1);  /* skip blank lines */
  if (sscanf(buf, INT32 " ! " TOKEN, &n, tok) != 2
      || strcasecmp(tok, "NBOND") != 0) {
    return ERROR(ERR_INPUT);  /* expecting NBOND keyword */
  }

  if ((s=Topology_setmaxnum_bond(t, n)) != OK) return ERROR(s);

  /* parse all bond pairs */
  for (i = 0;  i < n;  ) {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
    /*
     * format of bonds:  8 columns of ints, parse as pairs
     */
    for (j = 0;  j < NELEMS(bondfmt) && i < n;  j++, i++) {
      if (sscanf(buf, bondfmt[j], &bond.atomID[0], &bond.atomID[1]) < 2) {
        return ERROR(ERR_INPUT);  /* unable to parse bond pair */
      }
      bond.atomID[0]--;      /* adjust for C-based indexing */
      bond.atomID[1]--;
      bond.bondPrmID = -1;  /* not yet established */
      if ((s=Topology_add_bond(t, &bond)) != i) {
        return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
      }
    }
  }
  return OK;
}


int psf_read_angles(NL_FILE *f, Topology *t)
{
  char buf[BUFLEN];
  char tok[TOKLEN];
  Angle angle;
  int32 n, i, j, s;
  const char *anglefmt[] = {
    INT32 WS INT32 WS INT32,
    ISKIP WS ISKIP WS ISKIP WS INT32 WS INT32 WS INT32,
    ISKIP WS ISKIP WS ISKIP WS ISKIP WS ISKIP WS ISKIP WS
      INT32 WS INT32 WS INT32
  };

  if (Topology_angle_array_length(t) != 0) return ERROR(ERR_VALUE);

  /* parse NTHETA section */
  do {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
  } while (sscanf(buf, WS TOKEN, tok) != 1);  /* skip blank lines */
  if (sscanf(buf, INT32 " ! " TOKEN, &n, tok) != 2
      || strcasecmp(tok, "NTHETA") != 0) {
    return ERROR(ERR_INPUT);  /* expecting NTHETA keyword */
  }

  if ((s=Topology_setmaxnum_angle(t, n)) != OK) return ERROR(s);

  /* parse all angle triples */
  for (i = 0;  i < n;  ) {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
    /*
     * format of angles:  9 columns of ints, parse as triples
     */
    for (j = 0;  j < NELEMS(anglefmt) && i < n;  j++, i++) {
      if (sscanf(buf, anglefmt[j],
            &angle.atomID[0], &angle.atomID[1], &angle.atomID[2]) < 3) {
        return ERROR(ERR_INPUT);  /* unable to parse angle triple */
      }
      angle.atomID[0]--;       /* adjust for C-based indexing */
      angle.atomID[1]--;
      angle.atomID[2]--;
      angle.anglePrmID = -1;  /* not yet established */
      if ((s=Topology_add_angle(t, &angle)) != i) {
        return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
      }
    }
  }
  return OK;
}


int psf_read_dihedrals(NL_FILE *f, Topology *t)
{
  char buf[BUFLEN];
  char tok[TOKLEN];
  Dihed dihed;
  int32 n, i, j, s, dupcnt = 0;
  const char *dihedfmt[] = {
    INT32 WS INT32 WS INT32 WS INT32,
    ISKIP WS ISKIP WS ISKIP WS ISKIP WS INT32 WS INT32 WS INT32 WS INT32
  };

  if (Topology_dihed_array_length(t) != 0) return ERROR(ERR_VALUE);

  /* parse NPHI section */
  do {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
  } while (sscanf(buf, WS TOKEN, tok) != 1);  /* skip blank lines */
  if (sscanf(buf, INT32 " ! " TOKEN, &n, tok) != 2
      || strcasecmp(tok, "NPHI") != 0) {
    return ERROR(ERR_INPUT);  /* expecting NPHI keyword */
  }

  if ((s=Topology_setmaxnum_dihed(t, n)) != OK) return ERROR(s);

  /* parse all dihedral quadruples */
  for (i = 0;  i < n;  ) {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
    /*
     * format of dihedrals:  8 columns of ints, parse as quadruples
     */
    for (j = 0;  j < NELEMS(dihedfmt) && i < n;  j++, i++) {
      if (sscanf(buf, dihedfmt[j], &dihed.atomID[0], &dihed.atomID[1],
            &dihed.atomID[2], &dihed.atomID[3]) < 4) {
        return ERROR(ERR_INPUT);  /* unable to parse dihedral quadruple */
      }
      dihed.atomID[0]--;       /* adjust for C-based indexing */
      dihed.atomID[1]--;
      dihed.atomID[2]--;
      dihed.atomID[3]--;
      dihed.dihedPrmID = -1;  /* not yet established */
      if ((s=Topology_add_dihed(t, &dihed)) != i) {
        if (FAIL==s && Topology_getid_dihed(t,
              dihed.atomID[0], dihed.atomID[1],
              dihed.atomID[2], dihed.atomID[3]) == i-1) {
          /*
           * same as previous dihedral so don't list it again;
           * X-Plor PSF will repeat a dihedral quadruple corresponding
           * to its multiplicity
           */
          dupcnt++;  /* count number of duplicates */
          n--;       /* decrease total number of dihedral array elements */
          i--;       /* to compensate for loop increment */
        }
        else return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
      }
    }
  }
  if (dupcnt > 0) {
    /* shrink dihedral array from counting duplicates */
    if ((s=Array_setbuflen(&(t->dihed), n)) != OK) return ERROR(s);
  }
  return OK;
}


int psf_read_impropers(NL_FILE *f, Topology *t)
{
  char buf[BUFLEN];
  char tok[TOKLEN];
  Impr impr;
  int32 n, i, j, s;
  const char *imprfmt[] = {
    INT32 WS INT32 WS INT32 WS INT32,
    ISKIP WS ISKIP WS ISKIP WS ISKIP WS INT32 WS INT32 WS INT32 WS INT32
  };

  if (Topology_impr_array_length(t) != 0) return ERROR(ERR_VALUE);

  /* parse NIMPHI section */
  do {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
  } while (sscanf(buf, WS TOKEN, tok) != 1);  /* skip blank lines */
  if (sscanf(buf, INT32 " ! " TOKEN, &n, tok) != 2
      || strcasecmp(tok, "NIMPHI") != 0) {
    return ERROR(ERR_INPUT);  /* expecting NIMPHI keyword */
  }

  if ((s=Topology_setmaxnum_impr(t, n)) != OK) return ERROR(s);

  /* parse all improper quadruples */
  for (i = 0;  i < n;  ) {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
    /*
     * format of impropers:  8 columns of ints, parse as quadruples
     */
    for (j = 0;  j < NELEMS(imprfmt) && i < n;  j++, i++) {
      if (sscanf(buf, imprfmt[j], &impr.atomID[0], &impr.atomID[1],
            &impr.atomID[2], &impr.atomID[3]) < 4) {
        return ERROR(ERR_INPUT);  /* unable to parse improper quadruple */
      }
      impr.atomID[0]--;       /* adjust for C-based indexing */
      impr.atomID[1]--;
      impr.atomID[2]--;
      impr.atomID[3]--;
      impr.imprPrmID = -1;  /* not yet established */
      if ((s=Topology_add_impr(t, &impr)) != i) {
        return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
      }
    }
  }
  return OK;
}


int psf_read_donors(NL_FILE *f, Topology *t)
{
  char buf[BUFLEN];
  char tok[TOKLEN];
  const int32 natoms = Topology_atom_array_length(t);
  int32 n, i, j, s;
  int32 a, b;
  const char *pairfmt[] = {
    INT32 WS INT32,
    ISKIP WS ISKIP WS INT32 WS INT32,
    ISKIP WS ISKIP WS ISKIP WS ISKIP WS INT32 WS INT32,
    ISKIP WS ISKIP WS ISKIP WS ISKIP WS ISKIP WS ISKIP WS INT32 WS INT32
  };

  /* parse NDON section */
  do {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
  } while (sscanf(buf, WS TOKEN, tok) != 1);  /* skip blank lines */
  if (sscanf(buf, INT32 " ! " TOKEN, &n, tok) != 2
      || strcasecmp(tok, "NDON") != 0) {
    return ERROR(ERR_INPUT);  /* expecting NDON keyword */
  }

  /* parse all donor pairs, but don't store */
  for (i = 0;  i < n;  ) {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
    /*
     * format of donors:  8 columns of ints, parse as pairs
     */
    for (j = 0;  j < NELEMS(pairfmt) && i < n;  j++, i++) {
      if (sscanf(buf, pairfmt[j], &a, &b) < 2) {
        return ERROR(ERR_INPUT);  /* unable to parse donor pair */
      }
      if (a < 1 || a > natoms || b < 0 || b > natoms || a == b) {
        return ERROR(ERR_RANGE);
      }
    }
  }
  return OK;
}


int psf_read_acceptors(NL_FILE *f, Topology *t)
{
  char buf[BUFLEN];
  char tok[TOKLEN];
  const int32 natoms = Topology_atom_array_length(t);
  int32 n, i, j, s;
  int32 a, b;
  const char *pairfmt[] = {
    INT32 WS INT32,
    ISKIP WS ISKIP WS INT32 WS INT32,
    ISKIP WS ISKIP WS ISKIP WS ISKIP WS INT32 WS INT32,
    ISKIP WS ISKIP WS ISKIP WS ISKIP WS ISKIP WS ISKIP WS INT32 WS INT32
  };

  /* parse NACC section */
  do {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
  } while (sscanf(buf, WS TOKEN, tok) != 1);  /* skip blank lines */
  if (sscanf(buf, INT32 " ! " TOKEN, &n, tok) != 2
      || strcasecmp(tok, "NACC") != 0) {
    return ERROR(ERR_INPUT);  /* expecting NACC keyword */
  }

  /* parse all acceptor pairs, but don't store */
  for (i = 0;  i < n;  ) {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
    /*
     * format of acceptors:  8 columns of ints, parse as pairs
     */
    for (j = 0;  j < NELEMS(pairfmt) && i < n;  j++, i++) {
      if (sscanf(buf, pairfmt[j], &a, &b) < 2) {
        return ERROR(ERR_INPUT);  /* unable to parse acceptor pair */
      }
      if (a < 1 || a > natoms || b < 0 || b > natoms || a == b) {
        return ERROR(ERR_RANGE);
      }
    }
  }
  return OK;
}


int psf_read_exclusions(NL_FILE *f, Topology *t)
{
  char buf[BUFLEN];
  char tok[TOKLEN];
  Excl excl, *exclist;
  const int32 natoms = Topology_atom_array_length(t);
  int32 n, i, j, k, prev, next, s;
  const char *exclfmt[] = {
    INT32,
    ISKIP WS INT32,
    ISKIP WS ISKIP WS INT32,
    ISKIP WS ISKIP WS ISKIP WS INT32,
    ISKIP WS ISKIP WS ISKIP WS ISKIP WS INT32,
    ISKIP WS ISKIP WS ISKIP WS ISKIP WS ISKIP WS INT32,
    ISKIP WS ISKIP WS ISKIP WS ISKIP WS ISKIP WS ISKIP WS INT32,
    ISKIP WS ISKIP WS ISKIP WS ISKIP WS ISKIP WS ISKIP WS ISKIP WS INT32
  };
  Array arr;  /* array of Excl for staging input */

  if (Topology_excl_array_length(t) != 0) return ERROR(ERR_VALUE);

  /* parse NNB section */
  do {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
  } while (sscanf(buf, WS TOKEN, tok) != 1);  /* skip blank lines */
  if (sscanf(buf, INT32 " ! " TOKEN, &n, tok) != 2
      || strcasecmp(tok, "NNB") != 0) {
    return ERROR(ERR_INPUT);  /* expecting NNB keyword */
  }

  if ((s=Array_init(&arr, sizeof(Excl))) != OK) return ERROR(s);
  if ((s=Array_setbuflen(&arr, n)) != OK) return ERROR(s);

  /*
   * Rather than storing exclusions as pairs, they are stored in
   * a sparse-matrix-index format.  The first nexcl ints are the
   * second atom index of the exclusion pairs.  The next natoms
   * ints are a list of non-decreasing indices into the exclusion
   * list: the difference of a previous entry from the next entry
   * is the number of exclusions that that atom is to be included,
   * indexed from the previous entry to the next entry minus 1
   * (using zero-based indexing).
   *
   * Here is an example (from NAMD 1.5 source code comments):
   *
   *   3 !NNB
   *   3 4 5
   *   0 1 3 3 3
   *
   * This is a 5-atom system with 3 exclusions.  The exclusion list
   * generated is (2,3) (3,4) (3,5).  (Recall that since our data
   * arrays are C-based rather than Fortran-based, our exclusion
   * list pairs will actually be (1,2) (2,3) (2,4).)
   *
   * Note that although this seems like a strange way for storing
   * exclusion pairs, this will save space whenever the number of
   * explicit exclusions is greater than the number of atoms.
   */

  /* get exclusions (2nd atom of pair) */
  for (i = 0;  i < n;  ) {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      Array_done(&arr);
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
    /*
     * format of exclusion list:  8 columns of ints, parse as singletons
     */
    for (j = 0;  j < NELEMS(exclfmt) && i < n;  j++, i++) {
      if (sscanf(buf, exclfmt[j], &excl.atomID[1]) < 1) {
        Array_done(&arr);
        return ERROR(ERR_INPUT);  /* unable to parse exclusion */
      }
      /*
      if (excl.atomID[1] < 1 || excl.atomID[1] > natoms) {
        return ERROR("exclusion %d has out-of-range atom number %d",
            (int)i+1, (int)excl.atomID[1]);
      }
      */
      excl.atomID[1]--;     /* adjust for C-based indexing */
      excl.atomID[0] = -1;  /* not yet established */
      if ((s=Array_append(&arr, &excl)) != OK) {
        Array_done(&arr);
        return ERROR(s);
      }
    }
  }
  if (0 == n) {  /* must skip blank line */
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      Array_done(&arr);
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
    if (sscanf(buf, WS TOKEN, tok) == 1) {
      Array_done(&arr);
      return ERROR(ERR_INPUT);  /* expecting blank line */
    }
  }

  exclist = Array_data(&arr);   /* directly access exclusion array */

  /* next read in list of indices into exclusion array */
  for (prev = 0, i = 0;  i < natoms;  ) {
    if ((s=psf_get_line(f, buf, BUFLEN)) != OK) {
      Array_done(&arr);
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
    /*
     * format of exclusion list:  8 columns of ints, parse as singletons
     */
    for (j = 0;  j < NELEMS(exclfmt) && i < natoms;  j++, i++) {
      if (sscanf(buf, exclfmt[j], &next) < 1) {
        Array_done(&arr);
        return ERROR(ERR_INPUT);  /* unable to parse exclusion list index */
      }
      if (next < prev || next > n) {
        Array_done(&arr);
        return ERROR(ERR_INPUT);  /* exclusion list index has illegal value */
      }
      for (k = prev;  k < next;  k++) {
        exclist[k].atomID[0] = i;
        /* check validity */
        if (i == exclist[k].atomID[1]) {
          Array_done(&arr);
          return ERROR(ERR_INPUT);  /* exclusion has illegal value */
        }
      }
      prev = next;
    }
  }
  /* check validity */
  if (next != n) {
    Array_done(&arr);
    return ERROR(ERR_INPUT);  /* haven't initialized all exclusions */
  }

  /* store exclusions array into topology container */
  if ((s=Topology_setmaxnum_excl(t, n)) != OK) {
    Array_done(&arr);
    return ERROR(s);
  }
  for (i = 0;  i < n;  i++) {
    if ((s=Topology_add_excl(t, &exclist[i])) != i) {
      Array_done(&arr);
      return (s < FAIL ? ERROR(s) : ERROR(ERR_INPUT));
    }
  }
  Array_done(&arr);
  return OK;
}
