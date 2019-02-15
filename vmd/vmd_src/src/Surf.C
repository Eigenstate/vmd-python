/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: Surf.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.51 $	$Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   This file implements the VMD interface to the 'Surf' molecular surface
 *   compuatation program.
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Surf.h"
#include "Inform.h"
#include "utilities.h" // needed for vmd_getuid, vmd_delete_file, etc

// max length to allocate for filenames
#define VMD_FILENAME_MAX 1024

Surf::Surf() {}

int Surf::compute(float probe_r, int num, float *r, 
                   float *x, float *y, float *z) {
  FILE *outfile; // atom coords and radii we send to Surf
  FILE *infile;  // surface geometry we get back from Surf
  char *surfbin; // location of the Surf executable
  char *surfcmd; // Surf command string we actually execute
  static int surf_firsttime = 1; // flag used to print Surf citation

  if ((surfbin = getenv("SURF_BIN")) == NULL) {
    msgErr << "No SURF binary found; set the SURF_BIN environment variable"
           << sendmsg;
    msgErr << "to the location of your SURF executable." << sendmsg;
    return 0; // failure
  }

  //
  // construct the temp filenames we'll use for the coord/radii file 
  // we send to Surf and the triangulated surface file we get back
  //
  char *dirname = vmd_tempfile(""); 
  char *ofilename = new char[VMD_FILENAME_MAX];
  char *ifilename = new char[VMD_FILENAME_MAX];
  int rndnum = (vmd_random() % 999);
  sprintf(ofilename, "%svmdsurf.u%d.%d.in",     dirname, vmd_getuid(), rndnum);
  sprintf(ifilename, "%svmdsurf.u%d.%d.in.tri", dirname, vmd_getuid(), rndnum);
  delete [] dirname;
  vmd_delete_file(ofilename);
  vmd_delete_file(ifilename);

  //
  // write atom coordinates and radii to the file we send to Surf 
  //
  if ((outfile=fopen(ofilename, "wt")) == NULL) {
    msgErr << "Failed to create Surf atom radii input file" << sendmsg;
    if (ofilename) delete [] ofilename;			
    if (ifilename) delete [] ifilename;			
    return 0;  // failure
  }
  for (int i=0; i<num; i++) {
    fprintf(outfile, "%d %f %f %f %f\n", i, r[i], x[i], y[i], z[i]);
  }
  fclose(outfile);

  //
  // call Surf to calculate the surface for the given atoms
  //
  if ((surfcmd = new char[strlen(ofilename) + strlen(surfbin) + 82])) {
    sprintf(surfcmd, "\"%s\"  -W 1 -R %f %s", surfbin, probe_r, ofilename);
    vmd_system(surfcmd);    
    delete [] surfcmd;

    // print Surf citation the first time it is used in a VMD session
    if (surf_firsttime == 1) {
      surf_firsttime = 0;
      msgInfo << "This surface is made with SURF from UNC-Chapel Hill."
              << "  The reference is:" << sendmsg;
      msgInfo << "A. Varshney, F. P. Brooks, W. V. Wright, "
              << "Linearly Scalable Computation" << sendmsg;
      msgInfo << "of Smooth Molecular Surfaces, "
              << "IEEE Comp. Graphics and Applications, " << sendmsg;
      msgInfo << "v. 14 (1994) pp. 19-25." << sendmsg;
    }
  }
  
  // 
  // read the triangulated surface data 
  //
  if ((infile = fopen(ifilename, "r")) == NULL) {
    msgErr << "Cannot read SURF output file: " << ifilename << sendmsg;

    // Return cleanly, deleting temp files and so on. 
    vmd_delete_file(ofilename);				
    vmd_delete_file(ifilename);				
    if (ofilename) delete [] ofilename;			
    if (ifilename) delete [] ifilename;			
    return 0;  // failed
  }

  msgInfo << "Reading Surf geometry output file..." << sendmsg;
  numtriangles = 0; // no triangles read yet
  int vertnum = 0;  // no vertices read yet
  int atmindex;     // atom index this triangle goes with
  float vn[18];     // vertex and normal data

  // read in Surf geometry one triangle at a time
  // Each triangle is 18 floats, 3 x (3 coord values, 3 normal values)
  while (fscanf(infile, 
         "%d%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f", 
         &atmindex,
         vn,   vn+1,  vn+2, vn+3, vn+4, vn+5,
         vn+6, vn+7,  vn+8, vn+9,vn+10,vn+11,
         vn+12,vn+13,vn+14,vn+15,vn+16,vn+17) == 19) {
    if (!tri_degenerate(&vn[0], &vn[3], &vn[6])) {
      ind.append(atmindex); // add new atom index into triangle->atom map

      f.append(vertnum);    // add new vertex indices into facet list
      vertnum++;
      f.append(vertnum);    // add new vertex indices into facet list
      vertnum++;
      f.append(vertnum);    // add new vertex indices into facet list
      vertnum++;
      numtriangles++;       // total number of triangles added so far.

      // add new vertices and normals into vertex and normal lists
      v.append3x3(&vn[0], &vn[6], &vn[12]); // v0, v1, v2
      n.append3x3(&vn[3], &vn[9], &vn[15]); // n0, n1, n2
    }
  }
  fclose(infile); // file has been read in completely

  // Return cleanly, deleting temp files and so on.
  vmd_delete_file(ofilename);				
  vmd_delete_file(ifilename);				
  if (ifilename) delete [] ifilename;			
  if (ofilename) delete [] ofilename;			

  msgInfo << "Read Surf output file, processing geometry..." << sendmsg;

  return 1; // success
}

void Surf::clear() {
    numtriangles=0;
    v.clear();
    n.clear();
    f.clear();
  ind.clear();
}

