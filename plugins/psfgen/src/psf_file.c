#include <stdlib.h>
#include <string.h>

#include "psf_file.h"

#define PSF_RECORD_LENGTH 	160

/*
 * Read in the beginning of the bond/angle/dihed/etc information,
 * but don't read in the data itself.  Returns the number of the record type
 * for the molecule.  If error, returns (-1).
 */
int psf_start_block(FILE *file, const char *blockname) {
  char inbuf[PSF_RECORD_LENGTH+2];
  int nrec = -1;

  /* keep reading the next line until a line with blockname appears */
  do {
    if(inbuf != fgets(inbuf, PSF_RECORD_LENGTH+1, file)) {
      /* EOF encountered with no blockname line found ==> error, return (-1) */
      return (-1);
    }
    if(strlen(inbuf) > 0 && strstr(inbuf, blockname))
      nrec = atoi(inbuf);
  } while (nrec == -1);

  return nrec;
}


/* return # of atoms, or negative if error */
int psf_start_atoms(FILE *file) {
  char inbuf[PSF_RECORD_LENGTH+2];
  int natom = 0;
  
  /* skip comments; get number of atoms */
  /* Taken from VMD's ReadPSF */
  do {
    if (inbuf != fgets(inbuf, PSF_RECORD_LENGTH+1, file)) {
      /* EOF with no NATOM */
      return -1;  
    }
    if (strlen(inbuf) > 0) {
      if (!strstr(inbuf, "REMARKS")) {
        if (strstr(inbuf, "NATOM")) {
          natom = atoi(inbuf);
        }
      }
    }
  } while (!natom);
  return natom;
}


int psf_get_atom(FILE *f, char *name, char *atype, char *resname,
                 char *segname, char *resid, double *q, double *m) {

  char inbuf[PSF_RECORD_LENGTH+2];
  int i,num, read_count;

  if(inbuf != fgets(inbuf, PSF_RECORD_LENGTH+1, f)) {
    return(-1);
  }
  read_count = sscanf(inbuf, "%d %8s %8s %8s %8s %8s %lf %lf",
    &num, segname, resid, resname, name, atype, q, m);

  if (read_count != 8) {
    fprintf(stderr,"BAD ATOM LINE IN PSF FILE:\n: %s\n", inbuf);
    return -1;
  }
  if (sscanf(atype, "%d", &i) > 0) {
    fprintf(stderr, "PSF file is in CHARMM format; XPLOR format required.\n");
    return -1;
  }
  return num;
}
 

static int atoifw(char **ptr, int fw) {
  char *op = *ptr;
  int ival = 0;
  int iws = 0;
  char tmpc;

  sscanf(op, "%d%n", &ival, &iws);
  if ( iws == fw ) { /* "12345678 123..." or " 1234567 123..." */
    *ptr += iws;
  } else if ( iws < fw ) { /* left justified? */
    while ( iws < fw && op[iws] == ' ' ) ++iws;
    *ptr += iws;
  } else if ( iws < 2*fw ) { /* " 12345678 123..." */
    *ptr += iws;
  } else { /* " 123456712345678" or "1234567812345678" */
    tmpc = op[fw];  op[fw] = '\0';
    ival = atoi(op);
    op[fw] = tmpc;
    *ptr += fw;
  }
  return ival;
}


int psf_get_bonds(FILE *f, int fw, int n, int *bonds) {
  char inbuf[PSF_RECORD_LENGTH+2];
  char *bondptr = NULL;
  int i=0;
  while (i<n) {
    if((i % 4) == 0) {
      /* must read next line */
      if(!fgets(inbuf,PSF_RECORD_LENGTH+2,f)) {
        /* early EOF encountered */
        break;
      }
      bondptr = inbuf;
    }
    if((bonds[2*i] = atoifw(&bondptr,fw)) < 1)
      break;
    if((bonds[2*i+1] = atoifw(&bondptr,fw)) < 1)
      break;
    i++;
  }

  return (i != n);
}


int psf_get_angles(FILE *f, int fw, int n, int *angles) {
  char inbuf[PSF_RECORD_LENGTH+2];
  char *bondptr = NULL;
  int i=0;
  while (i<n) {
    if((i % 3) == 0) {
      /* must read next line */
      if(!fgets(inbuf,PSF_RECORD_LENGTH+2,f)) {
        /* early EOF encountered */
        break;
      }
      bondptr = inbuf;
    }
    if((angles[3*i] = atoifw(&bondptr,fw)) < 1)
      break;
    if((angles[3*i+1] = atoifw(&bondptr,fw)) < 1)
      break;
    if((angles[3*i+2] = atoifw(&bondptr,fw)) < 1)
      break;
    i++;
  }

  return (i != n);
}


int psf_get_dihedrals(FILE *f, int fw, int n, int *dihedrals) {
  char inbuf[PSF_RECORD_LENGTH+2];
  char *bondptr = NULL;
  int i=0;
  while (i<n) {
    if((i % 2) == 0) {
      /* must read next line */
      if(!fgets(inbuf,PSF_RECORD_LENGTH+2,f)) {
        /* early EOF encountered */
        break;
      }
      bondptr = inbuf;
    }
    if((dihedrals[4*i] = atoifw(&bondptr,fw)) < 1)
      break;
    if((dihedrals[4*i+1] = atoifw(&bondptr,fw)) < 1)
      break;
    if((dihedrals[4*i+2] = atoifw(&bondptr,fw)) < 1)
      break;
    if((dihedrals[4*i+3] = atoifw(&bondptr,fw)) < 1)
      break;
    i++;
  }

  return (i != n);
}


int psf_get_impropers(FILE *f, int fw, int n, int *impropers) {
  
  /* Same format */
  return psf_get_dihedrals(f, fw, n, impropers);
}


int psf_get_cmaps(FILE *f, int fw, int n, int *cmaps) {
  
  /* Same format */
  return psf_get_dihedrals(f, fw, 2*n, cmaps);
}

