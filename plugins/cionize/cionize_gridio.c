#if defined(_AIX)
/* Define to enable large file extensions on AIX */
#define _LARGE_FILE
#define _LARGE_FILES
#else
/* Defines which enable LFS I/O interfaces for large (>2GB) file support
 * on 32-bit machines.  These must be defined before inclusion of any
 * system headers.
 */
#define _LARGEFILE_SOURCE
#define _FILE_OFFSET_BITS 64
#endif

#include "cionize_gridio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h" /* timer code from Tachyon */

/* DX writing derived from VMD's dxplugin */
int write_grid_dx(const char *filename, int usebinary, float gridspacing, 
                  float cx, float cy, float cz,
                  int xsize, int ysize, int zsize,
                  float *datablock) {
  int i, j, k, count;
  const char *dataname = "cionize potential data";
  char *tmpstr, *p, *pp;
  const int xysize = xsize * ysize;
  const int total = xysize * zsize;
  double origin[3]={0, 0, 0};
  double xdelta[3]={1, 0, 0};
  double ydelta[3]={0, 1, 0};
  double zdelta[3]={0, 0, 1};
  FILE *fd;

  origin[0]=cx;
  origin[1]=cy;
  origin[2]=cz;
  xdelta[0]=gridspacing;
  ydelta[1]=gridspacing;
  zdelta[2]=gridspacing;

  fd = fopen(filename, "wb");

  fprintf(fd, "# Data from cionize 1.0\n#\n# Potential (kT/e at 298.15K)\n#\n");
  fprintf(fd, "object 1 class gridpositions counts %d %d %d\n",
          xsize, ysize, zsize);
  fprintf(fd, "origin %g %g %g\n",
          origin[0], origin[1], origin[2]);
  fprintf(fd, "delta %g %g %g\n",
          xdelta[0], xdelta[1], xdelta[2]);
  fprintf(fd, "delta %g %g %g\n",
          ydelta[0], ydelta[1], ydelta[2]);
  fprintf(fd, "delta %g %g %g\n",
          zdelta[0], zdelta[1], zdelta[2]);
  fprintf(fd, "object 2 class gridconnections counts %d %d %d\n",
          xsize, ysize, zsize);

  fprintf(fd, "object 3 class array type double rank 0 items %d %sdata follows\n", total, usebinary ? "binary " : "");
  count = 0;
  for (i=0; i<xsize; i++) {
      for (j=0; j<ysize; j++) {
          for (k=0; k<zsize; k++) {
              if (usebinary) {
                  fwrite(datablock + k*xysize + j*xsize + i, sizeof(float),
                          1, fd);
              } else {
                  fprintf(fd, "%g ", datablock[k*xysize + j*xsize + i]);
                  if (++count == 3) {
                      fprintf(fd, "\n");
                      count = 0;
                  }
              }
          }
      }
  }
  if (!usebinary && count)
    fprintf(fd, "\n");

  /* Remove quotes from dataname and print last line */
  tmpstr = malloc(sizeof(char) * (strlen(dataname)+1));
  strcpy(tmpstr, dataname);
  while ((p = strchr(tmpstr, '"')) != '\0') {
    pp = p+1;
    while ((*p++ = *pp++) != '\0') continue;
  }
  fprintf(fd, "object \"%s\" class field\n", tmpstr);
  free(tmpstr);

  fflush(fd);
  return 0;
}




int write_grid(const char* ofile, float gridspacing, float cx, float cy, float cz, long int numplane, long int numcol, long int numpt, const float* eners) {
  /* Write the current energy grid to a .dx file for later reading */
  FILE* dx_out;
  int i, j, k;
  int arrpos;
  int numentries;
  float starttime, endtime;
  rt_timerhandle timer;

  arrpos=0;
  dx_out = fopen(ofile, "w");

  if (dx_out ==  NULL) {
    fprintf(stderr, "Error: Couldn't open output dxfile %s. Exiting...", ofile);
    return 1;
  }

  timer = rt_timer_create();
  rt_timer_start(timer);

  /* start the timer */
  starttime = rt_timer_timenow(timer);

  /* Write a dx header */
  fprintf(dx_out, "# Data from cionize 1.0\n#\n# Potential (kT/e at 298.15K)\n#\n");
  fprintf(dx_out, "object 1 class gridpositions counts %li %li %li\n", numpt, numcol, numplane);
  fprintf(dx_out, "origin %12.6e %12.6e %12.6e\n", cx, cy, cz);
  fprintf(dx_out, "delta %12.6e %12.6e %12.6e\n", gridspacing, 0.0, 0.0);
  fprintf(dx_out, "delta %12.6e %12.6e %12.6e\n", 0.0, gridspacing, 0.0);
  fprintf(dx_out, "delta %12.6e %12.6e %12.6e\n", 0.0, 0.0, gridspacing);
  fprintf(dx_out, "object 2 class gridconnections counts %li %li %li\n", numpt, numcol, numplane);
  fprintf(dx_out, "object 3 class array type double rank 0 items %li data follows\n", numplane * numcol * numpt);

  /* Write the main data array */
  numentries = 0;
  for (i=0; i<numpt; i++) {
    for (j=0; j<numcol; j++) {
      for (k=0; k<numplane; k++) {
        arrpos = (k*numcol * numpt + j*numpt + i);
        fprintf(dx_out, "%-13.6e ", POT_CONV * eners[arrpos]);
        if (numentries % 3 == 2) fprintf(dx_out, "\n");
        numentries = numentries + 1;
      }
    }
  }

  /* Write the opendx footer */
  if (arrpos % 3 != 2) fprintf(dx_out, "\n");
  fprintf(dx_out, "attribute \"dep\" string \"positions\"\nobject \"regular positions regular connections\" class field\ncomponent \"positions\" value 1\ncomponent \"connections\" value 2\ncomponent \"data\" value 3");

  fclose(dx_out);

  /* check our time */
  endtime = rt_timer_timenow(timer);
  printf("Time for grid output: %.1f\n", endtime - starttime);
  rt_timer_destroy(timer);

  return 0;
}

int read_grid(const char* oldgridfile, float* grideners, float gridspacing, long int numplane, long int numcol, long int numpt, float cx, float cy, float cz) {
  /* Read an energy grid from a .dx file */
  int i, j, k;
  /* int arrpos; */
  char dataline[101];
  float tmpx, tmpy, tmpz;
  int count;
  float tmpspacing;
  FILE* dx_in;
  float starttime, endtime;
  rt_timerhandle timer;

  dx_in = fopen(oldgridfile, "r");
  if (dx_in == NULL) {
    fprintf(stderr, "Error: Couldn't open input dxfile %s. Exiting...", oldgridfile);
    return 1;
  }

  timer = rt_timer_create();
  rt_timer_start(timer);

  /* start the timer */
  starttime = rt_timer_timenow(timer);

  /* Read the header and check that the values are compatible */
  while (fgets(dataline,100,dx_in) && strncmp(dataline,"#",1) == 0);
  printf( "%s", dataline);
  sscanf(dataline, "%*s %*i %*s %*s %*s %i %i %i", &i, &j, &k);
  fgets(dataline, 100, dx_in);
  printf( "%s", dataline);
  sscanf(dataline, "%*s %e %e %e", &tmpx, &tmpy, &tmpz);
  fgets(dataline, 100, dx_in);
  printf( "%s", dataline);
  sscanf(dataline, "%*s %e %*e %*e\n", &tmpspacing);

  if (k != numplane || j != numcol || i != numpt || abs(tmpx - cx) > 0.01 || abs(tmpy - cy) > 0.01 || abs(tmpz - cz) > 0.01 || abs(gridspacing - tmpspacing) > 0.01) {
    fprintf(stderr, "Comparisons: %i/%li | %i/%li | %i/%li | %f/%f | %f/%f | %f/%f | %f/%f\n", i, numplane, j, numcol, k, numpt, tmpx, cx, tmpy, cy, tmpz, cz, gridspacing, tmpspacing);
    fprintf(stderr, "Error: Grid dimensions do not match those of the molecule in memory.\n");
    return 1;
  }

  /* Having passed those tests, skip four lines and start reading data*/
  fgets(dataline, 100, dx_in);
  fgets(dataline, 100, dx_in);
  fgets(dataline, 100, dx_in);
  fgets(dataline, 100, dx_in);


  for (count = 0; count < ((numplane * numcol * numpt)/3); count++) {
    fgets(dataline, 100, dx_in);
    sscanf(dataline, "%e %e %e", &grideners[transaddr(3*count, numplane, numcol, numpt)], &grideners[transaddr(3*count + 1, numplane, numcol, numpt)], &grideners[transaddr(3*count + 2, numplane, numcol, numpt)]);
  }

  if ((numplane * numcol * numpt) %3 != 0) {
    fgets(dataline, 100, dx_in);
    if ((numplane * numcol * numpt) %3 == 1) {
      sscanf(dataline, "%e", &grideners[transaddr(3*count, numplane, numcol, numpt)]);
    } else {
      sscanf(dataline, "%e %e", &grideners[transaddr(3*count, numplane, numcol, numpt)], &grideners[transaddr(3*count + 1, numplane, numcol, numpt)]);
    }
  }

  fclose(dx_in);

  /* Convert units to the proper internal units for cionize */
  for (i=0; i<numpt; i++) {
    for (j=0; j<numcol; j++) {
      for (k=0; k<numplane; k++) {
        float tmpnum;
        int arrpos;
        
        arrpos = (k*numcol * numpt + j*numpt + i);
        tmpnum = grideners[arrpos];
        tmpnum = tmpnum / POT_CONV;
        grideners[arrpos] = tmpnum;
      }
    }
  }

  /* check our time */
  endtime = rt_timer_timenow(timer);
  printf("Time for grid input: %.1f\n", endtime - starttime);
  rt_timer_destroy(timer);

  return 0;

}

int transaddr(const int count, const int numplane, const int numcol, const int numpt) {
  const int planesize = numcol * numplane;
  const int colsize = numplane;
  int i, j, k;
  int addr;
  int pos;

  i = count / planesize;
  pos = count % planesize;
  j = pos / colsize;
  k = pos % colsize;

  addr = (k * numcol * numpt + j * numpt + i);
  return addr;
}

