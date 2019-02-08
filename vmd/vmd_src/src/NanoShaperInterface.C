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
 *	$RCSfile: NanoShaperInterface.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.13 $	$Date: 2019/01/17 21:21:00 $
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


int NanoShaperInterface::compute_from_file(int surftype, float gspacing,
                                           float probe_radius,
                                           float skin_parm,
                                           float blob_parm, 
                                           int n, int *ids, float *xyzr, 
                                           int *flgs) {
  const char *nsbin = "NanoShaper";

  // generate output file name and try to create
  char *dirname = vmd_tempfile("");

  int uid = vmd_getuid();
  int rnd = vmd_random() % 999;

  char *filebase = new char[strlen(dirname) + 100];
  sprintf(filebase, "%svmdns.u%d.%d.", dirname, uid, rnd);
  delete [] dirname;

  char *pfilename = new char[strlen(filebase) + 100];
  char *ofilename = new char[strlen(filebase) + 100];
  sprintf(pfilename, "%sprm", filebase);
  sprintf(ofilename, "%sxyzr", filebase);

  FILE *pfile = fopen(pfilename, "wt");
  if (!pfile) {
    delete [] filebase;
    delete [] pfilename;
    delete [] ofilename;
    msgErr << "Failed to create NanoShaper parameter input file" << sendmsg;
    return 0;  // failure
  }

  const char *surfstr = "ses";
  const char *modestr = "normal";
  switch(surftype) {
    case NS_SURF_SES:
      surfstr = "ses";
      modestr = "normal";
      break;

    case NS_SURF_SKIN:
      surfstr = "skin";
      modestr = "normal";
      break;

    case NS_SURF_BLOBBY:
      surfstr = "blobby";
      modestr = "normal";
      break;

    case NS_SURF_POCKETS:
      surfstr = "pockets";
      modestr = "pockets";
      break;
  }

  //
  // Emit surface calculation configuration into NS input file
  // 
  fprintf(pfile, "Operative_Mode = %s\n", modestr);
  fprintf(pfile, "Pocket_Radius_Small = %.1f\n", 1.4);
  fprintf(pfile, "Pocket_Radius_Big = %.1f\n", 3.0);
  fprintf(pfile, "Surface = %s\n", surfstr);

  if (gspacing < 0.05f)
    gspacing = 0.05f;
  fprintf(pfile, "Grid_scale = %.1f\n", 1.0f/gspacing);

  fprintf(pfile, "XYZR_FileName = %s\n", ofilename);
  fprintf(pfile, "Probe_Radius = %.1f\n", probe_radius);
  fprintf(pfile, "Skin_Surface_Parameter = %.2f\n", skin_parm);
  fprintf(pfile, "Blobbyness = %.1f\n", blob_parm);

  // enable all to emulate MSMS
  fprintf(pfile, "Compute_Vertex_Normals = true\n");
  fprintf(pfile, "Save_Mesh_MSMS_Format = true\n");
  fprintf(pfile, "Vertex_Atom_Info = true\n");
 
  // various parameters copied from example input file
  fprintf(pfile, "Grid_perfil = %.1f\n", 70.0);
  fprintf(pfile, "Build_epsilon_maps = false\n");
  fprintf(pfile, "Build_status_map = true\n");
  fprintf(pfile, "Smooth_Mesh = true\n");
  fprintf(pfile, "Triangulation = true\n");

  // SD: optimized value for big structures
  fprintf(pfile, "Max_Probes_Self_Intersections = 5000\n");

  fprintf(pfile, "Self_Intersections_Grid_Coefficient = 5.0\n");
  fprintf(pfile, "Accurate_Triangulation = true\n");

  // Tell NS to store all files by appending to a base filename provided
  // by the caller, so we don't end up with problems on multi-user systems
  // or when multiple VMD sessions are running concurrently on the same
  // machine.  VMD chooses the base filename via OS temp file APIs.
  fprintf(pfile, "Root_FileName = %s\n", filebase);

  // SD: optimized value for big structures
  fprintf(pfile, "Max_skin_patches_per_auxiliary_grid_2d_cell = 2000\n");

  fclose(pfile);

  FILE *ofile = fopen(ofilename, "wt");
  if (!ofile) {
    delete [] ofilename;
    msgErr << "Failed to create NanoShaper atom xyzr input file" << sendmsg;
    return 0;  // failure
  }

  char *facetfilename = new char[strlen(filebase) + 100];
  char *vertfilename = new char[strlen(filebase) + 100];
  char *errfilename = new char[strlen(filebase) + 100];
  char *expfilename = new char[strlen(filebase) + 100];
  char *expindfilename = new char[strlen(filebase) + 100];
  char *areafilename = new char[strlen(filebase) + 100];
  sprintf(facetfilename, "%striangulatedSurf.face", filebase);
  sprintf(vertfilename, "%striangulatedSurf.vert", filebase);
  sprintf(errfilename, "%sstderror.txt", filebase);
  sprintf(expfilename, "%sexposed.xyz", filebase);
  sprintf(expindfilename, "%sexposedIndices.txt", filebase);
  sprintf(areafilename, "%striangleAreas.txt", filebase);

  // temporary files we want to make sure to clean up
  VMDTempFile ptemp(pfilename);
  VMDTempFile otemp(ofilename);
  VMDTempFile ftemp(facetfilename);
  VMDTempFile vtemp(vertfilename);
  VMDTempFile errtemp(errfilename);
  VMDTempFile exptemp(expfilename);
  VMDTempFile expindtemp(expindfilename);
  VMDTempFile areatemp(areafilename);

  //
  // write atom coordinates and radii to the file we send to NanoShaper 
  //
  for (int i=0; i<n; i++) {
    fprintf(ofile, "%f %f %f %f\n", 
            xyzr[4L*i], xyzr[4L*i+1], xyzr[4L*i+2], xyzr[4L*i+3]);
  }
  fclose(ofile);

  //
  // call NanoShaper to calculate the surface for the given atoms
  //
  char *nscmd = new char[2*strlen(ofilename) + strlen(nsbin) + 100];
  sprintf(nscmd, "\"%s\" %s", nsbin, pfilename);
  vmd_system(nscmd);    
  delete [] nscmd;
  delete [] pfilename;

  // 
  // read NanoShaper output files
  //
  if (surftype == NS_SURF_POCKETS) {
    // XXX pockets feature not complete yet
    msgErr << "NanoShaper pockets mode currently unimplemented" << sendmsg;
    return 0;
  } else {
    // Read output files for one of the normal surface modes

    // read facets
    FILE *facetfile = fopen(facetfilename, "r");
    if (!facetfile) {
      msgErr << "Cannot read NanoShaper facet file: " << facetfilename << sendmsg;
      return 0;  // failed
    }
    NanoShaperFace face;

    char trash[256];
    fgets(trash, sizeof(trash), facetfile); // eat text comments and counts
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
    fgets(trash, sizeof(trash), vertfile);
    fgets(trash, sizeof(trash), vertfile);
    fgets(trash, sizeof(trash), vertfile);
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
  } 


  return 1; // success
}


