/*
 * write DX grid file
 * $Id: dxfile.c,v 1.3 2005/07/20 15:37:39 johns Exp $
 *
 */
#include <stdio.h>

int write_dx_grid(FILE *file, const int *dims, const float *oddd,
		const float *data, float scale, const char *label) {

  int ii, jj, kk, ll;
  int ni, nj, nk, dj, dk, size;
  ni = dims[0];
  nj = dims[1];
  nk = dims[2];
  dj = dims[3];
  dk = dims[4];
  size = ni*nj*nk;

  fprintf(file, "# %s\n", label);
  fprintf(file, "object 1 class gridpositions counts %d %d %d\n", ni, nj, nk);
  fprintf(file, "origin %g %g %g\n", oddd[0], oddd[1], oddd[2]);
  fprintf(file, "delta %g %g %g\n",  oddd[3], oddd[4], oddd[5]);
  fprintf(file, "delta %g %g %g\n",  oddd[6], oddd[7], oddd[8]);
  fprintf(file, "delta %g %g %g\n",  oddd[9], oddd[10], oddd[11]);
  fprintf(file, "object 2 class gridconnections counts %d %d %d\n",
			ni, nj, nk);
  fprintf(file, "object 3 class array type double rank 0 items %d data follows\n", size);

  ll = 0;
  for (ii = 0; ii < ni; ++ii)
    for (jj = 0; jj < nj; ++jj)
      for (kk = 0; kk < nk; ++kk) {
        char c = ( ++ll % 3 && ll < size ? ' ' : '\n' );
        fprintf(file, "%g%c", scale*data[(ii*dj + jj)*dk + kk], c);
      }

  fprintf(file, "attribute \"dep\" string \"positions\"\n");
  fprintf(file, "object \"%s\" class field\n", label);
  fprintf(file, "component \"positions\" value 1\n");
  fprintf(file, "component \"connections\" value 2\n");
  fprintf(file, "component \"data\" value 3\n");

  return 0;
}

