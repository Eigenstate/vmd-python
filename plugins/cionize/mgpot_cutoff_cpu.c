#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "util.h"
#include "mgpot_defn.h"

/* #define SORT_BINS_CPU */

#undef  CHECK_CIRCLE
/* #define CHECK_CIRCLE */

#define CELLEN      4.f
#define INV_CELLEN  (1.f/CELLEN)

extern int cpu_compute_cutoff_potential_lattice(
    float *lattice,                    /* the lattice */
    int nx, int ny, int nz,            /* its dimensions, length nx*ny*nz */
    float xlo, float ylo, float zlo,   /* lowest corner of lattice */
    float gridspacing,                 /* lattice spacing */
    float cutoff,                      /* cutoff distance */
    Atom *atom,                        /* array of atoms */
    int natoms                         /* number of atoms */
    )
{
  const float a2 = cutoff * cutoff;
  const float a_1 = 1.f / cutoff;
  const float inv_a2 = a_1 * a_1;
  float s, gs;
  const float inv_gridspacing = 1.f / gridspacing;
  const int radius = (int) ceilf(cutoff * inv_gridspacing) - 1;
    /* lattice point radius about each atom */

  int n;
  int i, j, k;
  int ia, ib, ic;
  int ja, jb, jc;
  int ka, kb, kc;
  int index;
  int koff, jkoff;

  float x, y, z, q;
  float dx, dy, dz;
  float dz2, dydz2, r2;
  float e;
  float xstart, ystart;

  float *pg;

  int gindex;
  int ncell, nxcell, nycell, nzcell;
  int *first, *next;
  float inv_cellen = INV_CELLEN;
  float xmin, ymin, zmin;
  float xmax, ymax, zmax;

  /* find min and max extent */
  xmin = xmax = atom[0].x;
  ymin = ymax = atom[0].y;
  zmin = zmax = atom[0].z;
  for (n = 1;  n < natoms;  n++) {
    if (atom[n].x < xmin)      xmin = atom[n].x;
    else if (atom[n].x > xmax) xmax = atom[n].x;
    if (atom[n].y < ymin)      ymin = atom[n].y;
    else if (atom[n].y > ymax) ymax = atom[n].y;
    if (atom[n].z < zmin)      zmin = atom[n].z;
    else if (atom[n].z > zmax) zmax = atom[n].z;
  }

  /* number of cells in each dimension */
  nxcell = (int) floorf((xmax-xmin) * inv_cellen) + 1;
  nycell = (int) floorf((ymax-ymin) * inv_cellen) + 1;
  nzcell = (int) floorf((zmax-zmin) * inv_cellen) + 1;
  ncell = nxcell * nycell * nzcell;

  /* allocate for cursor link list implementation */
  first = (int *) malloc(ncell * sizeof(int));
  for (gindex = 0;  gindex < ncell;  gindex++) {
    first[gindex] = -1;
  }
  next = (int *) malloc(natoms * sizeof(int));
  for (n = 0;  n < natoms;  n++) {
    next[n] = -1;
  }

  /* geometric hashing */
  for (n = 0;  n < natoms;  n++) {
    if (0==atom[n].q) continue;  /* skip any non-contributing atoms */
    i = (int) floorf((atom[n].x - xmin) * inv_cellen);
    j = (int) floorf((atom[n].y - ymin) * inv_cellen);
    k = (int) floorf((atom[n].z - zmin) * inv_cellen);
    gindex = (k*nycell + j)*nxcell + i;
    next[n] = first[gindex];
    first[gindex] = n;
  }

#ifdef SORT_BINS_CPU
  /* sort atom bins using x-coordinate ordering */
  {
#define MAXBINSZ 20
    int indexbuf[MAXBINSZ];  /* assume upper bound on bin size */
    for (gindex = 0;  gindex < ncell;  gindex++) {
      /* copy index list into buffer */
      k = 0;
      for (n = first[gindex];  n != -1 && k < MAXBINSZ;  n = next[n], k++) {
        indexbuf[k] = n;
      }
      /* sort indices in buffer based on atom x-coordinate order */
      for (i = 1;  i < k;  i++) {
        n = indexbuf[i];
        j = i;
        while (j > 0 && atom[n].x > atom[ indexbuf[j-1] ].x) {
          /* sort in *decreasing* order, then list rebuilding reverses it */
          indexbuf[j] = indexbuf[j-1];
          j--;
        }
        indexbuf[j] = n;
      }
      /* rebuild list from sorted index buffer */
      first[gindex] = -1;
      for (i = 0;  i < k;  i++) {
        n = indexbuf[i];
        next[n] = first[gindex];
        first[gindex] = n;
      }
    }
  }
#ifdef SORT_BINS_CPU_DEBUG
  for (gindex = 0;  gindex < ncell;  gindex++) {
    printf("%d:", gindex);
    for (n = first[gindex];  n != -1;  n = next[n]) {
      printf(" %g", (double) atom[n].x);
    }
    printf("\n");
  }
#endif
#endif

  /* traverse the grid cells */
  for (gindex = 0;  gindex < ncell;  gindex++) {
    for (n = first[gindex];  n != -1;  n = next[n]) {
      x = atom[n].x - xlo;
      y = atom[n].y - ylo;
      z = atom[n].z - zlo;
      q = atom[n].q;

      /* find closest grid point with position less than or equal to atom */
      ic = (int) (x * inv_gridspacing);
      jc = (int) (y * inv_gridspacing);
      kc = (int) (z * inv_gridspacing);

      /* find extent of surrounding box of grid points */
      ia = ic - radius;
      ib = ic + radius + 1;
      ja = jc - radius;
      jb = jc + radius + 1;
      ka = kc - radius;
      kb = kc + radius + 1;

      /* trim box edges so that they are within grid point lattice */
      if (ia < 0)   ia = 0;
      if (ib >= nx) ib = nx-1;
      if (ja < 0)   ja = 0;
      if (jb >= ny) jb = ny-1;
      if (ka < 0)   ka = 0;
      if (kb >= nz) kb = nz-1;

      /* loop over surrounding grid points */
      xstart = ia*gridspacing - x;
      ystart = ja*gridspacing - y;
      dz = ka*gridspacing - z;
      for (k = ka;  k <= kb;  k++, dz += gridspacing) {
        koff = k*ny;
        dz2 = dz*dz;
#ifdef CHECK_CIRCLE_CPU
        /* note: enabling CHECK_CIRCLE_CPU with CHECK_CYLINDER_CPU
         * makes it a little slower */
        if (dz2 >= a2) continue;
#endif

        dy = ystart;
        for (j = ja;  j <= jb;  j++, dy += gridspacing) {
          jkoff = (koff + j)*nx;
          dydz2 = dy*dy + dz2;
#ifdef CHECK_CYLINDER_CPU
          if (dydz2 >= a2) continue;
#endif

          dx = xstart;
          index = jkoff + ia;
          pg = lattice + index;

#if defined(__INTEL_COMPILER)
          for (i = ia;  i <= ib;  i++, pg++, dx += gridspacing) {
            r2 = dx*dx + dydz2;
            s = r2 * inv_a2;
            gs = 1.875f + s*(-1.25f + s*0.375f);  /* TAYLOR2 */
            e = q * (1/sqrtf(r2) - a_1 * gs);
            *pg += (r2 < a2 ? e : 0);  /* LOOP VECTORIZED!! */
          }
#else
          for (i = ia;  i <= ib;  i++, pg++, dx += gridspacing) {
            r2 = dx*dx + dydz2;
            if (r2 >= a2) continue;
            s = r2 * inv_a2;
            gs = 1.875f + s*(-1.25f + s*0.375f);  /* TAYLOR2 */
            e = q * (1/sqrtf(r2) - a_1 * gs);
            *pg += e;
          }
#endif
        }
      } /* end loop over surrounding grid points */

    } /* end loop over atoms in a gridcell */
  } /* end loop over gridcells */

  /* free memory */
  free(next);
  free(first);

  return 0;
}
