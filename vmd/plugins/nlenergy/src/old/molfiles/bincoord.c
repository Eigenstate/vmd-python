/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 */

#include <stdlib.h>
#include <string.h>
#include "moltypes/const.h"
#include "moltypes/vecops.h"
#include "molfiles/bincoord.h"


#define MAXBUFSIZE  1024

static int32 read_natoms(NL_FILE *f);
static int read_coord(NL_FILE *f, dvec *, int32 natoms, BinCoordType);
static int write_natoms(NL_FILE *f, int32 natoms);
static int write_coord(NL_FILE *f, const dvec *, int32 natoms, BinCoordType);


int32 Bincoord_read_numatoms(const char *fname) {
  NL_FILE *f;
  int32 natoms;
  if (NULL==(f = NL_fopen(fname, "rb"))) return ERROR(ERR_FOPEN);
  else if ((natoms=read_natoms(f)) < 0) {
    NL_fclose(f);
    return ERROR(natoms);
  }
  else if (NL_fclose(f)) return ERROR(ERR_FCLOSE);
  return natoms;
}


int Bincoord_read(dvec *a, int32 natoms, BinCoordType atype,
    const char *fname) {
  NL_FILE *f;
  int s;
  if (NULL==(f = NL_fopen(fname, "rb"))) return ERROR(ERR_FOPEN);
  else if (natoms != read_natoms(f)) {
    NL_fclose(f);
    return (natoms >= 0 ? ERROR(ERR_VALUE) : ERROR(natoms));
  }
  else if ((s=read_coord(f, a, natoms, atype)) != OK) {
    NL_fclose(f);
    return ERROR(s);
  }
  else if (NL_fclose(f)) return ERROR(ERR_FCLOSE);
  return OK;
}


int Bincoord_write(const dvec *a, int32 natoms, BinCoordType atype,
    const char *fname) {
  NL_FILE *f;
  int s;
  if (NULL==(f = NL_fopen(fname, "wb"))) return ERROR(ERR_FOPEN);
  else if ((s=write_natoms(f, natoms)) != OK) {
    NL_fclose(f);
    return ERROR(s);
  }
  else if ((s=write_coord(f, a, natoms, atype)) != OK) {
    NL_fclose(f);
    return ERROR(s);
  }
  else if (NL_fclose(f)) return ERROR(ERR_FCLOSE);
  return OK;
}


int32 read_natoms(NL_FILE *f) {
  int32 natoms;
  if (NL_fread(&natoms, sizeof(int32), 1, f) != 1) return ERROR(ERR_FREAD);
  return natoms;
}


int read_coord(NL_FILE *f, dvec *a, int32 natoms, BinCoordType atype) {
  char c;
  if (natoms < 0) return ERROR(ERR_RANGE);
  else if (sizeof(dvec) != 3*sizeof(double)) {
    /* need to use intermediate buffer space */
    double buffer[3 * MAXBUFSIZE];
    int32 total, n = MAXBUFSIZE, i;
    for (total = 0;  total < natoms;  total += n) {
      n = (natoms - total >= MAXBUFSIZE ? MAXBUFSIZE : natoms - total);
      if (NL_fread(buffer, sizeof(double), 3*n, f) != 3*n) {
        return ERROR(ERR_FREAD);
      }
      for (i = 0;  i < n;  i++) {
        a[i].x = (dreal) buffer[3*i];
        a[i].y = (dreal) buffer[3*i+1];
        a[i].z = (dreal) buffer[3*i+2];
      }
    }
  }
  else {
    /* read directly to container */
    if (NL_fread(a, sizeof(dvec), natoms, f) != natoms) {
      return ERROR(ERR_INPUT);
    }
  }
  if (BINCOORD_NAMDVEL == atype) {  /* unit conversion */
    int32 i;
    for (i = 0;  i < natoms;  i++) {
      VECMUL(a[i], NAMD_VELOCITY_INTERNAL, a[i]);
    }
  }
  if (NL_fread(&c, sizeof(char), 1, f) != 0 || ! NL_feof(f)) {
    return ERROR(ERR_INPUT);
  }
  return OK;
}


int write_natoms(NL_FILE *f, int32 natoms) {
  if (natoms < 0) return ERROR(ERR_RANGE);
  if (NL_fwrite(&natoms, sizeof(int32), 1, f) != 1) return ERROR(ERR_FWRITE);
  return OK;
}


int write_coord(NL_FILE *f, const dvec *a, int32 natoms, BinCoordType atype) {
  ASSERT(natoms >= 0);
  if (BINCOORD_NAMDVEL == atype || sizeof(dvec) != 3*sizeof(double)) {
    /* need to use intermediate buffer space */
    double buffer[3 * MAXBUFSIZE];
    int32 total, n = MAXBUFSIZE, i;
    for (total = 0;  total < natoms;  total += n) {
      n = (natoms - total >= MAXBUFSIZE ? MAXBUFSIZE : natoms - total);
      for (i = 0;  i < n;  i++) {
        buffer[3*i] = (double) (a[i].x);
        buffer[3*i+1] = (double) (a[i].y);
        buffer[3*i+2] = (double) (a[i].z);
      }
      if (BINCOORD_NAMDVEL == atype) {  /* unit conversion */
        for (i = 0;  i < 3*n;  i++) {
          buffer[i] *= NAMD_VELOCITY_EXTERNAL;
        }
      }
      if (NL_fwrite(buffer, sizeof(double), 3*n, f) != 3*n) {
        return ERROR(ERR_FWRITE);
      }
    }
  }
  else {
    /* write directly from container */
    if (NL_fwrite(a, sizeof(dvec), natoms, f) != natoms) {
      return ERROR(ERR_FWRITE);
    }
  }
  return OK;
}
