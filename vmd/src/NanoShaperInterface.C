/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: NanoShaperInterface.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.3 $	$Date: 2015/05/31 22:53:21 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Start NanoSHaper and talk to it.
 *
 ***************************************************************************/

#include "utilities.h"
#include "vmdsock.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(ARCH_AIX4)
#include <strings.h>
#endif

#if defined(__irix)
#include <bstring.h>
#endif

#if defined(__hpux)
#include <time.h>
#endif

#include "NanoShaperInterface.h"
#include "Inform.h"


void NanoShaperInterface::clear() {
  atomids.clear();
  faces.clear();
  coords.clear();
  norms.clear();
}


// mark a file for destruction when the object goes out of scope
class VMDTempFile {
private:
  const char *m_filename;
public:
  VMDTempFile(const char *fname) {
    m_filename = stringdup(fname);
  }
  ~VMDTempFile() {
    vmd_delete_file(m_filename);
  }
};


int NanoShaperInterface::compute_from_file(int surftype,
                                           float probe_radius, float gspacing,
                                           int n, int *ids, float *xyzr, 
                                           int *flgs, int /* component */) {
  const char *nsbin = "NanoShaper";

  // generate output file name and try to create
  char *dirname = vmd_tempfile("");

  int uid = vmd_getuid();
  int rnd = vmd_random() % 999;

  char *pfilename = new char[strlen(dirname) + 100];
  char *ofilename = new char[strlen(dirname) + 100];
  sprintf(pfilename, "%svmdns.u%d.%d.prm", dirname, uid, rnd);
  sprintf(ofilename, "%svmdns.u%d.%d", dirname, uid, rnd);

  FILE *pfile = fopen(pfilename, "wt");
  if (!pfile) {
    delete [] pfilename;
    delete [] ofilename;
    msgErr << "Failed to create NanoShaper parameter input file" << sendmsg;
    return 0;  // failure
  }

  fprintf(pfile, "Operative_Mode = %s\n", "normal");
  fprintf(pfile, "Pocket_Radius_Big = %.1f\n", 3.0);
  fprintf(pfile, "Pocket_Radius_Small = %.1f\n", 1.4);

  const char *surfstr = "ses";
  switch(surftype) {
    case 0:
      surfstr = "ses";
      break;

    case 1:
      surfstr = "skin";
      break;

    case 2:
      surfstr = "blobby";
      break;
  }
  fprintf(pfile, "Surface = %s\n", surfstr);

  if (gspacing < 0.05f)
    gspacing = 0.05f;
  fprintf(pfile, "Grid_scale = %.1f\n", 1.0f/gspacing);

  fprintf(pfile, "XYZR_FileName = %s\n", ofilename);
  fprintf(pfile, "Probe_Radius = %.1f\n", probe_radius);
  fprintf(pfile, "Skin_Surface_Parameter = %.2f\n", 0.45);
  fprintf(pfile, "Blobbyness = %.1f\n", -2.5);

  // enable all to emulate MSMS
  fprintf(pfile, "Compute_Vertex_Normals = true\n");
  fprintf(pfile, "Save_Mesh_MSMS_Format = true\n");
  fprintf(pfile, "Vertex_Atom_Info = true\n");
 
  // various parameters copied from example input file
  fprintf(pfile, "Grid_perfill = %.1f\n", 70.0);
  fprintf(pfile, "Build_epsilon_maps = false\n");
  fprintf(pfile, "Build_status_map = true\n");
  fprintf(pfile, "Smooth_Mesh = true\n");
  fprintf(pfile, "Triangulation = true\n");
  fprintf(pfile, "Max_Probes_Self_Intersections = 100\n");
  fprintf(pfile, "Self_Intersections_Grid_Coefficient = 5.0\n");
  fprintf(pfile, "Accurate_Triangulation = true\n");
  fclose(pfile);

  delete [] dirname;


  FILE *ofile = fopen(ofilename, "wt");
  if (!ofile) {
    delete [] ofilename;
    msgErr << "Failed to create NanoShaper atom xyzr input file" << sendmsg;
    return 0;  // failure
  }

  char *facetfilename = new char[strlen(ofilename) + 6];
  char *vertfilename = new char[strlen(ofilename) + 6];
#if 1
  sprintf(facetfilename, "triangulatedSurf.face");
  sprintf(vertfilename, "triangulatedSurf.vert");
#else
  sprintf(facetfilename, "%s.face", ofilename);
  sprintf(vertfilename, "%s.vert", ofilename);
#endif

#if 1
  // temporary files we want to make sure to clean up
  VMDTempFile ptemp(pfilename);
  VMDTempFile otemp(ofilename);
  VMDTempFile ftemp(facetfilename);
  VMDTempFile vtemp(vertfilename);
#endif

  //
  // write atom coordinates and radii to the file we send to NanoShaper 
  //
  for (int i=0; i<n; i++) {
    fprintf(ofile, "%f %f %f %f\n", xyzr[4*i], xyzr[4*i+1], xyzr[4*i+2],
        xyzr[4*i+3]);
  }
  fclose(ofile);

  //
  // call NanoShaper to calculate the surface for the given atoms
  //
  {
    char *nscmd = new char[2*strlen(ofilename) + strlen(nsbin) + 100];
    sprintf(nscmd, "\"%s\" %s", nsbin, pfilename);
    vmd_system(nscmd);    
    delete [] nscmd;
  }
  delete [] pfilename;
  
  // read facets
  FILE *facetfile = fopen(facetfilename, "r");
  if (!facetfile) {
    msgErr << "Cannot read NanoShaper facet file: " << facetfilename << sendmsg;
    // Return cleanly, deleting temp files and so on. 
    return 0;  // failed
  }
  NanoShaperFace face;

  char trash[256];
  fgets(trash, sizeof(trash), facetfile);
  fgets(trash, sizeof(trash), facetfile);
  fgets(trash, sizeof(trash), facetfile);
  while (fscanf(facetfile, "%d %d %d %d %d",
        face.vertex+0, face.vertex+1, face.vertex+2, &face.surface_type,
        &face.anaface) == 5) {
    face.component = 0;  // XXX Unused by VMD, so why store?
    face.vertex[0]--;
    face.vertex[1]--;
    face.vertex[2]--;
    faces.append(face);
  }
  fclose(facetfile);

  msgInfo << "NanoShaper face count: " << faces.num() << sendmsg;

  // read verts
  FILE *vertfile = fopen(vertfilename, "r");
  if (!vertfile) {
    msgErr << "Cannot read NanoShaper vertex file: " << vertfilename << sendmsg;
    return 0;  // failed
  }
  NanoShaperCoord norm, coord;
  int atomid;  // 1-based atom index
  int l0fa;    // number of of the level 0 SES face
  int l;       // SES face level? (1/2/3)
  fgets(trash, sizeof(trash), facetfile);
  fgets(trash, sizeof(trash), facetfile);
  fgets(trash, sizeof(trash), facetfile);
  while (fscanf(vertfile, "%f %f %f %f %f %f %d %d %d",
        coord.x+0, coord.x+1, coord.x+2, 
        norm.x+0, norm.x+1, norm.x+2, 
        &l0fa, &atomid, &l) == 9) {
    norms.append(norm);
    coords.append(coord);
    atomids.append(atomid-1);
  }
  fclose(vertfile);

  msgInfo << "NanoShaper vert count: " << norms.num() << sendmsg;

  if (ids) {
    for (int i=0; i<atomids.num(); i++) {
      atomids[i] = ids[atomids[i]];
    }
  }
  return 1; // success
}


