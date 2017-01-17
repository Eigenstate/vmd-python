/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 */

#define PDBATOM_CREATION_TIME

#include <stdlib.h>
#include <string.h>

#ifdef PDBATOM_CREATION_TIME
#include <time.h>
#endif

#include "moltypes/const.h"
#include "moltypes/vecops.h"
#include "molfiles/pdbatom.h"


int PdbAtom_init(PdbAtom *p) {
  int s;
  if ((s=Array_init(&(p->atomCoord), sizeof(dvec))) != OK) return ERROR(s);
  if ((s=Array_init(&(p->pdbAtomAux), sizeof(PdbAtomAux))) != OK) {
    return ERROR(s);
  }
  return OK;
}

void PdbAtom_done(PdbAtom *p) {
  Array_done(&(p->atomCoord));
  Array_done(&(p->pdbAtomAux));
}

int PdbAtom_setup(PdbAtom *p, int32 n) {
  int s;
  if (n < 0) return ERROR(ERR_VALUE);
  else if (n > 0) {
    if ((s=Array_resize(&(p->atomCoord), n)) != OK) return ERROR(s);
    if ((s=Array_resize(&(p->pdbAtomAux), n)) != OK) return ERROR(s);
  }
  return OK;
}

int32 PdbAtom_numatoms(const PdbAtom *p) {
  ASSERT(Array_length(&(p->atomCoord)) == Array_length(&(p->pdbAtomAux)));
  return Array_length(&(p->atomCoord));
}

const dvec *PdbAtom_coord_const(const PdbAtom *p) {
  return Array_data_const(&(p->atomCoord));
}

const PdbAtomAux *PdbAtom_aux_const(const PdbAtom *p) {
  return Array_data_const(&(p->pdbAtomAux));
}

dvec *PdbAtom_coord(PdbAtom *p) {
  return Array_data(&(p->atomCoord));
}

PdbAtomAux *PdbAtom_aux(PdbAtom *p) {
  return Array_data(&(p->pdbAtomAux));
}

int PdbAtom_set_coord(PdbAtom *p, const dvec *a, int32 n) {
  if (n != Array_length(&(p->atomCoord))) return ERROR(ERR_VALUE);
  memcpy(Array_data(&(p->atomCoord)), a, n*sizeof(dvec));
  return OK;
}

int PdbAtom_set_aux(PdbAtom *p, const PdbAtomAux *a, int32 n) {
  if (n != Array_length(&(p->pdbAtomAux))) return ERROR(ERR_VALUE);
  memcpy(Array_data(&(p->pdbAtomAux)), a, n*sizeof(PdbAtomAux));
  return OK;
}


static int reader(NL_FILE *f, PdbAtom *p, PdbAtomType atype);
static int writer(NL_FILE *f, const PdbAtom *p, PdbAtomType atype);

int PdbAtom_read(PdbAtom *p, PdbAtomType atype, const char *fname) {
  NL_FILE *f;
  int s;
  if (NULL==(f = NL_fopen(fname, "r"))) return ERROR(ERR_FOPEN);
  else if ((s=reader(f, p, atype)) != OK) {
    NL_fclose(f);
    return ERROR(s);
  }
  else if (NL_fclose(f)) return ERROR(ERR_FCLOSE);
  return OK;
}

int PdbAtom_write(const PdbAtom *p, PdbAtomType atype, const char *fname) {
  NL_FILE *f;
  int s;
  if (NULL==(f = NL_fopen(fname, "w"))) return ERROR(ERR_FOPEN);
  else if ((s=writer(f, p, atype)) != OK) {
    NL_fclose(f);
    return ERROR(s);
  }
  else if (NL_fclose(f)) return ERROR(ERR_FCLOSE);
  return OK;
}


static int pdbatom_get_line(NL_FILE *f, char *line, int len);
static int pdbatom_decode(dvec *v, PdbAtomAux *a, const char *buf);
static int pdbatom_encode(char *buf, const dvec *v, const PdbAtomAux *a);

/* PDB_LINELEN set by specification, BUFLEN must be two chars larger */
#define BUFLEN       84
#define PDB_LINELEN  80


int reader(NL_FILE *f, PdbAtom *p, PdbAtomType atype) {
  dvec *coord = Array_data(&(p->atomCoord));
  PdbAtomAux *aux = Array_data(&(p->pdbAtomAux));
  char line[BUFLEN];
  int32 len;
  int32 cnt = 0;
  const int32 nexpect = Array_length(&(p->atomCoord));
  int s;

  ASSERT(Array_length(&(p->pdbAtomAux)) == nexpect);

  while ((len = pdbatom_get_line(f, line, BUFLEN)) > 0) {
    if (len > PDB_LINELEN+1) {
      return ERROR(ERR_INPUT);  /* line in PDB file is too long */
    }
    if (strncmp(line, "ATOM  ", 6) != 0 && strncmp(line, "HETATM", 6) != 0) {
      continue;  /* skip anything that isn't an atom record */
    }
    len--;  /* remove trailing newline */
    while (len < PDB_LINELEN) {
      line[len] = ' ';  /* pad short lines with spaces */
      len++;
    }
    line[len] = '\0';  /* replace string terminator */

    if (0 == nexpect) {
      dvec v;
      PdbAtomAux a;
      if ((s=pdbatom_decode(&v, &a, line)) != OK) return ERROR(s);
      if (PDBATOM_VEL == atype) {
        VECMUL(v, PDBATOM_VELOCITY_INTERNAL, v);
      }
      if ((s=Array_append(&(p->atomCoord), &v)) != OK) return ERROR(s);
      if ((s=Array_append(&(p->pdbAtomAux), &a)) != OK) return ERROR(s);
    }
    else if (cnt < nexpect) {
      if ((s=pdbatom_decode(&coord[cnt], &aux[cnt], line)) != OK) {
        return ERROR(s);
      }
      if (PDBATOM_VEL == atype) {
        VECMUL(coord[cnt], PDBATOM_VELOCITY_INTERNAL, coord[cnt]);
      }
    }
    else {
      return ERROR(ERR_INPUT); /* found more atom records than expected */
    }
    cnt++;
  }
  if (len != 0) {
    return ERROR(ERR_INPUT);  /* pdbatom_get_line() failed */
  }
  else if (cnt < nexpect) {
    return ERROR(ERR_INPUT);  /* found fewer atom records than expected */
  }
  return OK;
}


int pdbatom_get_line(NL_FILE *f, char *line, int len) {
  ASSERT(len >= 3);
  if (NL_fgets(line, len, f) == NULL) {
    line[0] = '\0';
    if ( ! NL_feof(f)) return ERROR(ERR_FREAD);
  }
  return strlen(line);
}


int writer(NL_FILE *f, const PdbAtom *p, PdbAtomType atype) {
  const dvec *coord = Array_data_const(&(p->atomCoord));
  const PdbAtomAux *aux = Array_data_const(&(p->pdbAtomAux));
  const int32 natoms = Array_length(&(p->atomCoord));
  int32 i;
  char line[BUFLEN];
  int s;
#ifdef PDBATOM_CREATION_TIME
  struct tm *tmbuf;
  time_t abstime;
  int n;

  /* print REMARK record regarding creation */
  abstime = time(NULL);
  tmbuf = localtime(&abstime);
  if (NULL == tmbuf
      || (n = strftime(line, BUFLEN, "REMARK Created %d %b %Y "
          "at %H:%M by NAMD-Lite PDB atoms file writer\n", tmbuf)) == 0
      || n > PDB_LINELEN+1 || n >= 0) {
    snprintf(line, BUFLEN, "REMARK Created by NAMD-Lite "
        "PDB atoms file writer\n");
  }
#else
  snprintf(line, BUFLEN, "REMARK Created by NAMD-Lite PDB atoms file writer\n");
#endif
  if (NL_fputs(line, f) < 0) {
    return ERROR(ERR_FWRITE);  /* NL_fputs() failed for REMARK */
  }

  ASSERT(Array_length(&(p->pdbAtomAux)) == natoms);

  for (i = 0;  i < natoms;  i++) {
    if (PDBATOM_VEL == atype) {
      dvec sv;
      VECMUL(sv, PDBATOM_VELOCITY_EXTERNAL, coord[i]);
      if ((s=pdbatom_encode(line, &sv, &aux[i])) != OK) return ERROR(s);
    }
    else {
      if ((s=pdbatom_encode(line, &coord[i], &aux[i])) != OK) return ERROR(s);
    }
    if (NL_fputs(line, f) < 0) {
      return ERROR(ERR_FWRITE);  /* NL_fputs() failed for atom record */
    }
  }

  /* print END record */
  if (NL_fputs("END\n", f) < 0) {
    return ERROR(ERR_FWRITE);  /* NL_fputs() failed for END */
  }
  return OK;
}



/******************************************************************************
 *
 * low level parsing of ATOM and HETATM records from PDB file
 *
 ******************************************************************************/

typedef struct Cols_t {
  int scol, ecol;  /* start, end columns */
} Cols;

static const Cols atomCols[] = {
  /*
   * ATOM and HETATM record format as given by
   * http://www.rcsb.org/pdb/docs/format/pdbguide2.2/guide2.2_frame.html
   */
/* columns     data type     field       definition                          */
/* ------------------------------------------------------------------------- */
  { 1,  6}, /* record name   "ATOM  "    (or "HETATM")                       */
  { 7, 11}, /* Integer       serial      Atom serial number.                 */
  {13, 16}, /* Atom          name        Atom name.                          */
  {17, 17}, /* Character     altLoc      Alternate location indicator.       */
  {18, 20}, /* Residue name  resName     Residue name.                       */
  {22, 22}, /* Character     chainID     Chain identifier.                   */
  {23, 26}, /* Integer       resSeq      Residue sequence number.            */
  {27, 27}, /* AChar         iCode       Code for insertion of residues.     */
  {31, 38}, /* Real(8.3)     x           Orthogonal coordinates for X.       */
  {39, 46}, /* Real(8.3)     y           Orthogonal coordinates for Y.       */
  {47, 54}, /* Real(8.3)     z           Orthogonal coordinates for Z.       */
  {55, 60}, /* Real(6.2)     occupancy   Occupancy.                          */
  {61, 66}, /* Real(6.2)     tempFactor  Temperature factor.                 */
  {73, 76}, /* LString(4)    segID       Segment identifier (left-justified) */
  {77, 78}, /* LString(2)    element     Element symbol (right-justified)    */
  {79, 80}, /* LString(2)    charge      Charge on the atom.                 */
};


/* used when reading to parse line */
int pdbatom_decode(dvec *v, PdbAtomAux *a, const char *buf)
{
  int start, len;
  const Cols *c;
  char s_x[12], s_y[12], s_z[12];
  char s_occupancy[8], s_tempFactor[8];
  char ch;

  ASSERT(strlen(buf) == PDB_LINELEN);

  c = &atomCols[0];
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(a->record) > len);
  strncpy(a->record, &buf[start], len);
  a->record[len] = '\0';

  c++;
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(a->serial) > len);
  strncpy(a->serial, &buf[start], len);
  a->serial[len] = '\0';

  c++;
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(a->name) > len);
  strncpy(a->name, &buf[start], len);
  a->name[len] = '\0';

  c++;
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(a->altLoc) > len);
  strncpy(a->altLoc, &buf[start], len);
  a->altLoc[len] = '\0';

  c++;
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(a->resName) > len);
  strncpy(a->resName, &buf[start], len);
  a->resName[len] = '\0';

  c++;
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(a->chainID) > len);
  strncpy(a->chainID, &buf[start], len);
  a->chainID[len] = '\0';

  c++;
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(a->resSeq) > len);
  strncpy(a->resSeq, &buf[start], len);
  a->resSeq[len] = '\0';

  c++;
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(a->iCode) > len);
  strncpy(a->iCode, &buf[start], len);
  a->iCode[len] = '\0';

  c++;
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(s_x) > len);
  strncpy(s_x, &buf[start], len);
  s_x[len] = '\0';
  if (sscanf(s_x, FMT_DREAL "%c", &(v->x), &ch) != 1) {
    return ERROR(ERR_INPUT);  /* failed to find x-coordinate while parsing */
  }

  c++;
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(s_y) > len);
  strncpy(s_y, &buf[start], len);
  s_y[len] = '\0';
  if (sscanf(s_y, FMT_DREAL "%c", &(v->y), &ch) != 1) {
    return ERROR(ERR_INPUT);  /* failed to find y-coordinate while parsing */
  }

  c++;
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(s_z) > len);
  strncpy(s_z, &buf[start], len);
  s_z[len] = '\0';
  if (sscanf(s_z, FMT_DREAL "%c", &(v->z), &ch) != 1) {
    return ERROR(ERR_INPUT);  /* failed to find z-coordinate while parsing */
  }

  c++;
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(s_occupancy) > len);
  strncpy(s_occupancy, &buf[start], len);
  s_occupancy[len] = '\0';
  if (sscanf(s_occupancy, FMT_FREAL "%c", &(a->occupancy), &ch) != 1) {
    return ERROR(ERR_INPUT);  /* failed to find occupancy while parsing */
  }

  c++;
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(s_tempFactor) > len);
  strncpy(s_tempFactor, &buf[start], len);
  s_tempFactor[len] = '\0';
  if (sscanf(s_tempFactor, FMT_FREAL "%c", &(a->tempFactor), &ch) != 1) {
    return ERROR(ERR_INPUT);  /* failed to find temperature factor */
  }

  c++;
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(a->segID) > len);
  strncpy(a->segID, &buf[start], len);
  a->segID[len] = '\0';

  c++;
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(a->element) > len);
  strncpy(a->element, &buf[start], len);
  a->element[len] = '\0';

  c++;
  start = c->scol - 1;
  len = c->ecol - c->scol + 1;
  ASSERT(sizeof(a->charge) > len);
  strncpy(a->charge, &buf[start], len);
  a->charge[len] = '\0';

  return OK;
}


/* used when writing to formulate line */
int pdbatom_encode(char *buf, const dvec *v, const PdbAtomAux *a)
{
  int n;
#ifdef DEBUG_SUPPORT
  const Cols *c = &atomCols[0];
  ASSERT(strlen(a->record) == c->ecol - c->scol + 1);
  c++;
  ASSERT(strlen(a->serial) == c->ecol - c->scol + 1);
  c++;
  ASSERT(strlen(a->name) == c->ecol - c->scol + 1);
  c++;
  ASSERT(strlen(a->altLoc) == c->ecol - c->scol + 1);
  c++;
  ASSERT(strlen(a->resName) == c->ecol - c->scol + 1);
  c++;
  ASSERT(strlen(a->chainID) == c->ecol - c->scol + 1);
  c++;
  ASSERT(strlen(a->resSeq) == c->ecol - c->scol + 1);
  c++;
  ASSERT(strlen(a->iCode) == c->ecol - c->scol + 1);
  c += 6;
  ASSERT(strlen(a->segID) == c->ecol - c->scol + 1);
  c++;
  ASSERT(strlen(a->element) == c->ecol - c->scol + 1);
  c++;
  ASSERT(strlen(a->charge) == c->ecol - c->scol + 1);
#endif
  /* make sure numeric ranges are correct */
  if (v->x < -999.999 || v->x > 9999.999) {
    return ERROR(ERR_VALUE);  /* x-coordinate out-of-range while writing */
  }
  if (v->y < -999.999 || v->y > 9999.999) {
    return ERROR(ERR_VALUE);  /* y-coordinate out-of-range while writing */
  }
  if (v->z < -999.999 || v->z > 9999.999) {
    return ERROR(ERR_VALUE);  /* z-coordinate out-of-range while writing */
  }
  if (a->occupancy < -99.99 || a->occupancy > 999.99) {
    return ERROR(ERR_VALUE);  /* occupancy out-of-range while writing */
  }
  if (a->tempFactor < -99.99 || a->tempFactor > 999.99) {
    return ERROR(ERR_VALUE);  /* temperature factor out-of-range */
  }

  /* write fields into buffer, make sure line length is correct */
  n = snprintf(buf, PDB_LINELEN+2,
      "%s%s %s%s%s %s%s%s   %8.3f%8.3f%8.3f%6.2f%6.2f      %s%s%s\n",
      a->record, a->serial, a->name, a->altLoc, a->resName, a->chainID,
      a->resSeq, a->iCode, (double) v->x, (double) v->y, (double) v->z,
      (double) a->occupancy, (double) a->tempFactor,
      a->segID, a->element, a->charge);
  if (n != PDB_LINELEN+1 || buf[PDB_LINELEN] != '\n') {
    return ERROR(ERR_RANGE);  /* incorrect line length for record */
  }
  ASSERT(strlen(buf) == PDB_LINELEN+1);
  return OK;
}
