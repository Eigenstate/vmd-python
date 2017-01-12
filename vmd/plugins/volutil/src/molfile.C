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
 *	$RCSfile: molfile.C,v $
 *	$Author: ltrabuco $	$Locker:  $		$State: Exp $
 *	$Revision: 1.2 $	$Date: 2009/08/25 19:50:16 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 *
 ***************************************************************************/

#include <stdio.h>
#include <string.h>
#include <errno.h>

//#include "libmolfile_plugin.h"
//#include "molfile_plugin.h"
#include "getplugins.h"

#include "volmap.h"

/* Initialize the hash and plugin arrays */
void init_plugins();

static bool plugins_inited = false;

// Read in a map file
int VolMap::load (const char *filename) {

  char maptype_str[80];
  int i;
  
  if (!refname) {
    char *tmpstr = new char[strlen(filename) + 3];
    sprintf(tmpstr, "\"%s\"", filename);
    set_refname(tmpstr);
    delete[] tmpstr;
  }
  //printf("%s <- \"%s\"\n", get_refname(), filename);
  
  // 1. Load the proper molfile plugin for the provided map.
  
  if (!plugins_inited) init_plugins();
  plugins_inited = true;
  
  molfile_plugin_t* plugin = get_plugin_forfile(filename, maptype_str);

  if (!plugin || !(plugin->read_volumetric_data)) {
    //fprintf(stderr, "ERROR: Can't read the input file of type \"%s\".\n", maptype_str);
    fprintf(stderr, "ERROR: Can't read the input file %s.\n", filename);
    return 1;
  }


  // 2. Read in the data
  
  int natom;
  int setsinfile = 0;
  molfile_volumetric_t *metadata;
  float *datablock=NULL, *colorblock=NULL;

  //printf("Loading \"%s\"\n", filename);

  void* maphandler = plugin->open_file_read(filename, maptype_str, &natom);
  if (!maphandler) {
    fprintf(stderr, "ERROR: Unable to read input file. Exiting...\n");
    return 1;
  }

  // Fetch metadata from file
  plugin->read_volumetric_metadata(maphandler, &setsinfile, &metadata);

#if defined(DEBUG)
  printf("DEBUG: setsinfile = %d\n", setsinfile);
#endif

  // Get first dataset
  const int setid = 0;

  xsize =  metadata[setid].xsize;
  ysize =  metadata[setid].ysize;
  zsize =  metadata[setid].zsize;
  int mapsize = xsize*ysize*zsize;
  
  if (data) delete[] data;
  
  datablock = new float[mapsize];
  data = new float[mapsize];
    
  if (metadata[setid].has_color)
    colorblock = new float[3*mapsize];
  if (plugin->read_volumetric_data(maphandler, setid, datablock, colorblock)) {
    printf("Error reading volumetric data set %d\n", setid);
    delete[] datablock;
  }
  
  if (colorblock) delete[] colorblock;
  
  for (i=0; i<mapsize; i++)
    data[i] = datablock[i];   // float to double
  delete[] datablock;
  
  for (i=0; i<3; i++) {
    origin[i] = metadata[setid].origin[i];
    xaxis[i] = metadata[setid].xaxis[i];
    yaxis[i] = metadata[setid].yaxis[i];
    zaxis[i] = metadata[setid].zaxis[i];
    xdelta[i] = xaxis[i]/(xsize - 1);
    ydelta[i] = yaxis[i]/(ysize - 1);
    zdelta[i] = zaxis[i]/(zsize - 1);
  }

  dataname = new char[strlen(metadata[setid].dataname)+1];
  strcpy(dataname, metadata[setid].dataname);
  char *p = NULL;
  p = strstr(dataname, "pmf [kT],");
  if (p) sscanf(p, "pmf [kT], %lf frames, T = %lf Kelvin, %*s",
		&weight, &temperature);


#if defined(DEBUG)
  printf("DEBUG: dataname = %s\n", metadata[setid].dataname);
  printf("DEBUG: origin = %8g %8g %8g\n", metadata[setid].origin[0], metadata[setid].origin[1], metadata[setid].origin[2]);
  printf("DEBUG: xaxis  = %8g %8g %8g\n", metadata[setid].xaxis[0], metadata[setid].xaxis[1], metadata[setid].xaxis[2]);
  printf("DEBUG: yaxis  = %8g %8g %8g\n", metadata[setid].yaxis[0], metadata[setid].yaxis[1], metadata[setid].yaxis[2]);
  printf("DEBUG: zaxis  = %8g %8g %8g\n", metadata[setid].zaxis[0], metadata[setid].zaxis[1], metadata[setid].zaxis[2]);
  printf("DEBUG: xsize  = %d\n", xsize);
  printf("DEBUG: ysize  = %d\n", ysize);
  printf("DEBUG: zsize  = %d\n", zsize);
  printf("DEBUG: has_color = %d\n", metadata[setid].has_color);
  printf("DEBUG: temperature = %.2f\n", temperature);
  printf("DEBUG: weight = %.2f\n", weight);
#endif

  return 0; 
}




// Write volmap contents to a DX file
int VolMap::write_old (const char *filename) const {
  if (!data) return -1;
  int gridsize = xsize*ysize*zsize;
  int i;
  
  printf("%s -> \"%s\"\n", get_refname(), filename);
  
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



int VolMap::write (const char *filename) const {

  int i;

  if (!data) return -1;
  printf("%s -> \"%s\"\n", get_refname(), filename);
  char maptype_str[80];

  // Load the proper molfile plugin for the provided map.
  
  if (!plugins_inited) init_plugins();
  plugins_inited = true;

  molfile_plugin_t* plugin = get_plugin_forfile(filename, maptype_str);

  if (!plugin || !(plugin->write_volumetric_data)) {
    fprintf(stderr, "ERROR: Can't write output map file %s.\n", filename);
    return 1;
  }

  int natom = 0;
  molfile_volumetric_t *metadata;
  float *datablock=NULL, *colorblock=NULL;

  void* filehandler = plugin->open_file_write(filename, maptype_str, natom);
  if (!filehandler) {
    fprintf(stderr, "ERROR: Unable to write ouput file. Exiting...\n");
    return 1;
  }

  // Copy data to datablock
  int mapsize = xsize*ysize*zsize;
  datablock = new float[mapsize];
  for (i=0; i<mapsize; i++)
    datablock[i] = data[i]; // double to float

  // Set metadata
  int setsinfile = 1;
  metadata = new molfile_volumetric_t[setsinfile];

  char name[256];
  sprintf(name, "created by volutil, T = %.2f Kelvin,", temperature);
  for (int setid=0; setid<setsinfile; setid++) {
    strcpy(metadata[setid].dataname, name);
    metadata[setid].xsize = xsize;
    metadata[setid].ysize = ysize;
    metadata[setid].zsize = zsize;
    metadata[setid].has_color = false;
    for (i=0; i<3; i++) {
      metadata[setid].origin[i] = origin[i];
      metadata[setid].xaxis[i] = xaxis[i];
      metadata[setid].yaxis[i] = yaxis[i];
      metadata[setid].zaxis[i] = zaxis[i];
    }
  }

  if (plugin->write_volumetric_data(filehandler, metadata, datablock, colorblock) != MOLFILE_SUCCESS)
    fprintf(stderr, "Failed to write volumetric data.\n");
  plugin->close_file_write(filehandler);

  delete [] datablock;

  return 0;
}


