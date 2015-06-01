/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2009 The Board of Trustees of the
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: io.C,v $
 *	$Author: ltrabuco $	$Locker:  $		$State: Exp $
 *	$Revision: 1.1 $	$Date: 2009/08/06 20:58:45 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * This file contains a DX file reader/write code for the VolMap class. 
 * 
 * This code is presently unused since we are now using the VMD plugins for 
 * loading volumetric maps. However, io.C can be swapped with molfile.C at 
 * any time to revert to DX-only loading. Doing so removes all dependences on 
 * external libraries (VMD molfile, Tcl, netcdfm, libdl, etc.), which can be
 * useful for those who are in a hurry to compile volutil...
 *
 * XXX - The above comment is most likely not true anymore. This file
 *       should probably go away soon.
 *
 ***************************************************************************/

#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "volmap.h"



// Get a string from a stream, printing any errors that occur
char *dxgets(char *s, int n, FILE *stream) {
  char *returnVal;

  if (feof(stream)) {
    fprintf(stderr, "Unexpected end-of-file.\n");
    return NULL;
  } else if (ferror(stream)) {
    fprintf(stderr, "Error reading file.\n");
    return NULL;
  } else {
    returnVal = fgets(s, n, stream);
    if (returnVal == NULL) {
      fprintf(stderr, "Error reading line (error = %d).\n", errno);
    }
  }

  return returnVal;
}


// Read in a DX file
int VolMap::load (const char *filename) {
  const int LINESIZE=512;
  char inbuf[LINESIZE];
  int i;
  
  printf("%s: Loading DX map from file \"%s\"\n", refname, filename);
  
  FILE *fd = fopen(filename, "r");
  if (!fd) {
    fprintf(stderr, "Error: Could not open file \"%s\" for reading\n", filename);
    return -1;
  };
  
  /* skip comments */
//  do {
//    if (!dxgets(inbuf, LINESIZE, fd)) return -1;
//  } while (inbuf[0] == '#' && strncmp(inbuf, "# VMDTAG", 8));
//
  /* get the VMD weight */
//  if (sscanf(inbuf, "# VMDTAG WEIGHT %lg", &weight) != 1) {
//    weight = 1.;
//    fprintf(stderr, "No weight detected in DX file.\n");
//  }   
  
  do {
    if (!dxgets(inbuf, LINESIZE, fd)) return -1;
  } while (inbuf[0] == '#');
              
  /* get the number of grid points */
  if (sscanf(inbuf, "object 1 class gridpositions counts %i %i %i", &xsize, &ysize, &zsize) != 3) {
    fprintf(stderr, "Error reading grid dimensions.\n");
    return -1;
  }

  /* get the cell origin */
  if (dxgets(inbuf, LINESIZE, fd) == NULL) {
    return -1;
  }
  if (sscanf(inbuf, "origin %lf %lf %lf", origin, origin+1, origin+2) != 3) {
    fprintf(stderr, "Error reading grid origin.\n");
    return -1;
  }

  /* get the cell dimensions */
  if (dxgets(inbuf, LINESIZE, fd) == NULL) return -1;
  if (sscanf(inbuf, "delta %lf %lf %lf", xdelta, xdelta+1, xdelta+2) != 3) {
    fprintf(stderr, "Error reading cell x-dimension.\n");
    return -1;
  }

  if (dxgets(inbuf, LINESIZE, fd) == NULL) return -1;
  if (sscanf(inbuf, "delta %lf %lf %lf", ydelta, ydelta+1, ydelta+2) != 3) {
    fprintf(stderr, "Error reading cell y-dimension.\n");
    return -1;
  }

  if (dxgets(inbuf, LINESIZE, fd) == NULL) return -1;
  
  if (sscanf(inbuf, "delta %lf %lf %lf", zdelta, zdelta+1, zdelta+2) != 3) {
    fprintf(stderr, "Error reading cell z-dimension.\n");
    return -1;
  }

  /* skip the last two lines of the header*/
  if (dxgets(inbuf, LINESIZE, fd) == NULL) return -1;
  if (dxgets(inbuf, LINESIZE, fd) == NULL) return -1;

  /* Set the unit cell origin and basis vectors */
  for (i=0; i<3; i++) {
    xaxis[i] = xdelta[i] * (xsize-1);
    yaxis[i] = ydelta[i] * (ysize-1);
    zaxis[i] = zdelta[i] * (zsize-1);
  }  
  
  /* Read the values from the file */
  int xysize = xsize*ysize;
  int gridsize = xsize*ysize*zsize;
  int gx, gy, gz; 
  double grid[3];
  int count;
    
  if (data) delete[] data;
  data = new double[gridsize];
      
  gx = gy = gz = 0;  
  for (count=0; count < gridsize/3; count++) {
    if (dxgets(inbuf, LINESIZE, fd) == NULL ) return -1;
    
    if (sscanf(inbuf, "%lf %lf %lf", grid, grid+1, grid+2) != 3) {
      fprintf(stderr, "Error reading grid data.\n");
      return -1;
    }
  
    for (i=0; i < 3; i++) { 
      data[gx + gy*xsize + gz*xysize] = grid[i];
      gz++;
      if (gz >= zsize) {
        gz = 0;
        gy++;
        if (gy >= ysize) {
          gy = 0;
          gx++;
        }
      }
    }
    
  }

  // This reads the last data line, if it only contains 1 or 2 voxels
  if (gridsize%3) {
    if (dxgets(inbuf, LINESIZE, fd) == NULL )
      return -1;

    count = sscanf(inbuf, "%lf %lf %lf", grid, grid+1, grid+2);
    if (count != (gridsize%3)) {
      fprintf(stderr, "Error: incorrect number of data points.\n");
      return -1;
    }

    for (i=0; i<count; i++) {
      data[gx + gy*xsize + gz*xysize] = grid[i];
      gz++;
    }
  }
  
  fclose(fd);
  
  printf("%s: %d voxels\n", refname, gridsize);
  
  return 0;
   
}




// Write volmap contents to a DX file
int VolMap::write (const char *filename) const {
  if (!data) return -1;
  int gridsize = xsize*ysize*zsize;
  int i;
  
  printf("%s: Writing DX map to file \"%s\"\n", refname, filename);
  
  FILE *fout = fopen(filename, "w");
  if (!fout) {
    fprintf(stderr, "volmap: Error: Cannot open file \"%s\" for writing\n", filename);
    return -1;
  };
    
  fprintf(fout, "# Data calculated by volutil (http://www.ks.uiuc.edu)\n");
  fprintf(fout, "# VMDTAG WEIGHT %g\n", weight);
 
  fprintf(fout, "object 1 class gridpositions counts %d %d %d\n", xsize, ysize, zsize);
  fprintf(fout, "origin %g %g %g\n", origin[0], origin[1], origin[2]);
  fprintf(fout, "delta %g %g %g\n", xdelta[0], xdelta[1], xdelta[2]);
  fprintf(fout, "delta %g %g %g\n", ydelta[0], ydelta[1], ydelta[2]);
  fprintf(fout, "delta %g %g %g\n", zdelta[0], zdelta[1], zdelta[2]);
  fprintf(fout, "object 2 class gridconnections counts %d %d %d\n", xsize, ysize, zsize);
  fprintf(fout, "object 3 class array type double rank 0 items %d data follows\n", gridsize);
  
  // This reverses the ordering from x fastest to z fastest changing variable
  float val1,val2,val3;
  int gx=0, gy=0, gz=-1;
  for (i=0; i < (gridsize/3)*3; i+=3)  {
    if (++gz >= zsize) {
      gz=0;
      if (++gy >= ysize) {gy=0; gx++;}
    }
    val1 = voxel_value(gx,gy,gz);
    if (++gz >= zsize) {
      gz=0;
      if (++gy >= ysize) {gy=0; gx++;}
    }
    val2 = voxel_value(gx,gy,gz);
    if (++gz >= zsize) {
      gz=0;
      if (++gy >= ysize) {gy=0; gx++;}
    }
    val3 = voxel_value(gx,gy,gz);    
    fprintf(fout, "%g %g %g\n", val1, val2, val3);
  }
  for (i=(gridsize/3)*3; i < gridsize; i++) {
    if (++gz >= zsize) {
      gz=0;
      if (++gy >= ysize) {gy=0; gx++;}
    }
    fprintf(fout, "%g ", voxel_value(gx,gy,gz));
  }
  if (gridsize%3) fprintf(fout, "\n");
  
  fprintf(fout, "\n");

  // XXX todo: make sure that "dataname" contains no quotes
  fprintf(fout, "object \"%s\" class field\n", "volutil output");
  
  fclose(fout);
  return 0;
}
