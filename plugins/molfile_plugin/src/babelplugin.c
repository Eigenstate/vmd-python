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
 *      $RCSfile: babelplugin.c,v $
 *      $Author: johns $       $Locker:  $             $State: Exp $
 *      $Revision: 1.47 $       $Date: 2009/04/29 15:45:27 $
 *
 ***************************************************************************/

/*
 * Convert files using Babel 1.6
 *   http://www.eyesopen.com/products/applications/babel.html
 *
 * Convert files using Open Babel 1.100.2
 *   http://openbabel.sourceforge.net/babel.shtml
 */


#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#if !defined(_MSC_VER) 
#include <unistd.h>  /* for getuid */
#endif

#include "molfile_plugin.h"
#include "readpdb.h"
#include "vmddir.h"
#include "periodic_table.h"

typedef struct {
  FILE *fd;
  int natoms;
  char *original_file;
  char *current_file;
  int babel_num;
  int babel_i;
} pdbdata;

/* 
 * I guess this is to try to keep tmp files from clobbering each other
 */
static int vmd_getuid(void) {
#if defined(_MSC_VER)
  return 0;
#else
  return getuid();
#endif
}

static int vmd_delete_file(const char * path) {
#if defined(_MSC_VER)
  if (DeleteFile(path) == 0)
    return -1;
  else
    return 0;
#else
  return unlink(path);
#endif
}

#define BABEL_TMPDIR "/tmp/"

/*
 * Dude, don't even think for a minute that I came up with this code.  It's
 * copied from BabelConvert.C, ok?
 * This gets called three times, with has_multi = 1, -2, and -1.  Don't ask
 * me why.
 */
static char *file(const char *filename, int idx, int has_multi) {
   /* temp space to save the filename or glob */
   int i=0;
   char *ptr;
   char *tempspace = (char *)malloc(513);
   const char *s;
   for (s = filename; *s != 0; s++) { 
      if ((*s == '/') || (*s == '\\'))
        i = s-filename+1;
   }
   /*
   // so filename+i points to the actual name
   // if there are multiple files in the conversion, then the output
   // looks like "test0041.extensions".  If there was a single file,
   // the output looks like "test.extensions"
   */
   if (has_multi == -1) {
      sprintf(tempspace, "%svmdbabel*.u%d.%s", BABEL_TMPDIR,
              vmd_getuid(), filename + i);
   } else if (has_multi == -2) {
      char *reallytemp = (char *)malloc(strlen(filename+i)+1);
      strcpy(reallytemp, filename+i);
      *(reallytemp + strlen(reallytemp) - 1) = 0;
      sprintf(tempspace, "vmdbabel%%[0-9.]u%d.%s%%c",
              vmd_getuid(), reallytemp);
      free(reallytemp);
   } else if (has_multi == 0) {
      sprintf(tempspace, "%svmdbabel.u%d.%s", BABEL_TMPDIR,
              vmd_getuid(), filename + i);
   } else {
      sprintf(tempspace, "%svmdbabel%04d.u%d.%s", BABEL_TMPDIR, idx+1,
              vmd_getuid(), filename + i);
   }
   for (ptr = tempspace; *ptr; ptr++) {  /* babel makes them lowercase! */
      *ptr = tolower(*ptr);              /* grrrrrrr                    */
   }
   return tempspace;
}

static void delete_all(const char *filename) {
  const char *s;
  char *t;

  s = file(filename, 0, -1); /* puts a '*' in number field */
  t = (char *)malloc(strlen(s) + 35);
#if defined(_MSC_VER)
   sprintf(t, "del %s", s);
#else
   sprintf(t, "/bin/rm -f \"%s\"", s);
#endif
  system(t);
  free(t);
}
 
static void *open_pdb_read(const char *filepath, int *natoms) {
  FILE *fd;
  pdbdata *pdb;
  char pdbstr[PDB_BUFFER_LENGTH];
  int indx;
  fd = fopen(filepath, "r");
  if (!fd) return NULL;
  pdb = (pdbdata *)malloc(sizeof(pdbdata));
  pdb->fd = fd;
  *natoms = 0;
  do {
    if((indx = read_pdb_record(pdb->fd, pdbstr)) == PDB_ATOM)
      *natoms += 1;
  } while (indx != PDB_END && indx != PDB_EOF);
  rewind(pdb->fd);
  pdb->natoms = *natoms;
  return pdb;
}

/*
 * Babel 1.6 internal file type names
 */
static const char *babel16filetypes[] = {
"alc",
"prep",
"bs",
"bgf",
"car",
"boog",
"caccrt",
"cadpac",
"charmm",
"c3d1",
"c3d2",
"cssr",
"fdat",
"gstat",
"dock",
"dpdb",
"feat",
"fract",
"gamout",
"gzmat",
"gauout",
"g94",
"gr96A",
"gr96N",
"hin",
"sdf",
"m3d",
"macmol",
"macmod",
"micro",
"mm2in",
"mm2out",
"mm3",
"mmads",
"mdl",
"molen",
"mopcrt",
"mopint",
"mopout",
"pcmod",
"psin",
"psout",
"msf",
"schakal",
"shelx",
"smiles",
"spar",
"semi",
"spmm",
"mol",
"mol2",
"wiz",
"unixyz",
"xyz",  
"xed",
0
};

/*
 * Plugin names registered in VMD for each Babel 1.6 file type
 */
static const char *babel16filetypenames[] = {
  "Alchemy",          "AMBERPREP",       "BallStick",      
  "MSIBGF",           "BiosymCAR",       "Boogie",
  "Cacao",            "CADPAC",          "CHARMm",
  "Chem3d-1",         "Chem3d-2",        "CSSR",
  "FDAT",             "GSTAT",           "Dock",
  "DockPDB",          "Feature",         "Fractional",    
  "GAMESSoutput",     "GaussianZmatrix", "Gaussian92output", 
  "Gaussian94output", "Gromos96A",       "Gromos96N",
  "HyperchemHIN",     "IsisSDF",         "M3D",
  "MacMolecule",      "Macromodel",      "MicroWorld",
  "MM2Input",         "MM2Output",       "MM3",
  "MMADS",            "MDLMOL",          "MOLIN",
  "MopacCartesian",   "MopacInternal",   "MopacOutput",
  "PCModel",          "PSGVBin",         "PSGVBout",
  "QuantaMSF",        "Schakal",         "ShelX",
  "SMILES",
  "Spartan",          "SpartanSE",       "SpartanMM",
  "SybylMol",         "SybylMol2",       "Conjure",
  "UniChemXYZ",       "XYZ",             "XED", 
  0
};


/*
 * Open Babel 1.100.2 internal file type names
 */
static const char *openbabel11filetypes[] = {
"alc",
"prep",
"bs",
"caccrt",
"ccc",
"c3d1",
"c3d2",
"cml",
"crk2d",
"crk3d",
"box",
"dmol",
"feat",
"gam",
"gpr",
"mm1gp",
"qm1gp",
"hin",
"jout",
"bin",
"mmd",
"car",
"sdf",
"mol",
"mopcrt",
"mopout",
"mmads",
"mpqc",
"bgf",
"nwo",
"pqs",
"qcout",
"res",
"smi",
"mol2",
"unixyz",
"vmol",
"xyz",
0
};

/*
 * Plugin names registered in VMD for each Open Babel 1.100.2 file type
 */
static const char *openbabel11filetypenames[] = {
  "Alchemy",          "AMBERPREP",       "BallStick",      
  "Cacao",            "CCC",           
  "Chem3d-1",         "Chem3d-2",        "ChemicalMarkup"
  "CRK2D",            "CRK3D",           "Dock35Box",
  "Dmol3Coord",       "Feature",         "GAMESSoutput",     
  "GhemicalProj",     "GhemicalMM",      "GhemicalQM",
  "HyperchemHIN",     "JaguarOutput",    "OpenEyeBinary",
  "Macromodel",       "BiosymCAR",       "IsisSDF",
  "MDLMOL",           "MopacCartesian",  "MopacOutput",     
  "MMADS",            "MPQC",            "MSIBGF",  
  "NWChemOutput",     "PQS",             "QChemOutput",
  "ShelX",            "SMILES",          "SybylMol2",   
  "UniChemXYZ",       "ViewMol",         "XYZ",
  0
};


static const char *babel16type_from_name(const char *name) {
  const char **ptr = babel16filetypenames;
  int i=0; 
  while (*ptr) {
    if (!strcmp(*ptr, name))
      return babel16filetypes[i];
    ptr++;
    i++;
  }
  return NULL;
}

static const char *openbabel11type_from_name(const char *name) {
  const char **ptr = openbabel11filetypenames;
  int i=0; 
  while (*ptr) {
    if (!strcmp(*ptr, name))
      return openbabel11filetypes[i];
    ptr++;
    i++;
  }
  return NULL;
}


/* 
 * Figure out the file type, call babel, and return a handle if successful.
 * From this point we're just reading in a pdb file.
 */
static void *open_babel_read(const char *filename, const char *filetypename,
    int *natoms) {

  const char *babelbin;
  char *current_file;
  pdbdata *pdb;
  char *s;
  const char *fmt;
  int count = 0;
  VMDDIR *dirp;
  char *dp;
  char temps[100];
  char tempc;
  char lastc;
  int may_have_multi = 0;
  char *tmp_multi = NULL;
  const char *filetype;

  babelbin = getenv("VMDBABELBIN");
  if (!babelbin) {
    fprintf(stderr, "Babel plugin needs VMDBABELBIN environment variable\n"
                    "to point to location of Babel executable\n");
    return NULL;
  }

#if 0
  /* Try Open Babel file type names first... */ 
  filetype = openbabel11type_from_name(filetypename);
  if (!filetype) {
    fprintf(stderr, "No Open Babel 1.100.2 file type for '%s'\n", filetypename);
  }
#endif

  /* Try Babel 1.6 file type names if Open Babel didn't match */ 
  filetype = babel16type_from_name(filetypename);
  if (!filetype) {
    fprintf(stderr, "No Babel 1.6 file type for '%s'\n", filetypename);
    return NULL;
  }
  s = (char *)malloc(strlen(babelbin) +               
              strlen(" -i       -opdb ") +
              strlen(filename) +
              strlen(file(filename, 0, 1)) +
              20);
 
  /*
  // On windows its necessary to quote command names due to
  // the high tendency for paths to have spaces in them.
  */
  sprintf(s, "\"%s\" -i%s \"%s\" all -opdb \"%s\"",
     babelbin, filetype, filename, (const char *)file(filename, 0, 0));

  delete_all(filename);       /* delete any conflicting existing files  */
  system(s);                  /* run the babel command                  */
  free(s);

  /* now find how many frames were printed */
  fmt = file(filename, 0, -2);
  dirp = vmd_opendir(BABEL_TMPDIR);
  if (dirp == NULL) {
    return NULL; /* failure */
  }
 
   lastc = *(filename + strlen(filename) -1);

   while ((dp = vmd_readdir(dirp)) != NULL) {
      if (sscanf(dp, fmt, temps, &tempc) > 1 && lastc == tempc) {
     count++;
     /* check if there is 1 element but Babel thinks there are several */
     if (count == 1) {
        if (strstr(dp, "0001.")) {
           may_have_multi = 1;
           tmp_multi = strdup(dp);
        }
     }
      }
   }
   vmd_closedir(dirp);

   if (may_have_multi && count == 1) {
      /* then move the test0001.extension file to test.extension */
      char *s2, *t2;
      s2 = (char *)malloc(2*(strlen(tmp_multi)+strlen(BABEL_TMPDIR))+40);

#if defined(_MSC_VER)
      sprintf(s2, "move \"%s\\%s\" \"%s\\\"", BABEL_TMPDIR, tmp_multi, BABEL_TMPDIR);
#else
      sprintf(s2, "mv \"%s/%s\" \"%s/\"", BABEL_TMPDIR, tmp_multi, BABEL_TMPDIR);
#endif

      t2 = strstr(tmp_multi, "0001.");
      *t2 = 0;
      strcat(s2, tmp_multi);
      strcat(s2, t2 + 4);
      fprintf(stderr, "%s\n", s2);
      system(s2);
      free(s2);
   }

   if (tmp_multi) {
      free(tmp_multi);
   }

  /*
   * Ok, now that we're done with all that crap, we should have a bunch
   * of temp files.  Now we need to open the first one to get the
   * number of atoms.
   */
  if (count == 0) {
    fprintf(stderr, "Babel molecule file translation failed!\n");
    return NULL;
  }
  current_file = file(filename, 0, count > 1);
  pdb = open_pdb_read(current_file, natoms);
  if (!pdb) {
    fprintf(stderr, "Couldn't read structure from Babel pdb output\n");
    free(current_file);
    return NULL;
  }
  pdb->original_file = strdup(filename); 
  pdb->current_file = current_file;
  pdb->babel_num = count;
  pdb->babel_i = 1;
  return pdb;
}

static int read_pdb_structure(void *mydata, int *optflags, 
    molfile_atom_t *atoms) {
  pdbdata *pdb = (pdbdata *)mydata;
  molfile_atom_t *atom;
  char pdbrec[PDB_BUFFER_LENGTH];
  int i, rectype, atomserial, pteidx;
  char ridstr[8];
  char elementsymbol[3];
  int badptecount = 0;
  long fpos = ftell(pdb->fd);
 
  *optflags = MOLFILE_INSERTION | MOLFILE_OCCUPANCY | MOLFILE_BFACTOR | 
              MOLFILE_ALTLOC | MOLFILE_ATOMICNUMBER;

  i = 0;
  do {
    rectype = read_pdb_record(pdb->fd, pdbrec);
    switch (rectype) {
    case PDB_ATOM:
      atom = atoms+i;
      get_pdb_fields(pdbrec, strlen(pdbrec), &atomserial,
          atom->name, atom->resname, atom->chain, atom->segid, 
          ridstr, atom->insertion, atom->altloc, elementsymbol,
          NULL, NULL, NULL, &atom->occupancy, &atom->bfactor);
      atom->resid = atoi(ridstr);

      /* determine atomic number from the element symbol */
      pteidx = get_pte_idx_from_string(elementsymbol);
      atom->atomicnumber = pteidx;
      if (pteidx != 0) {
        atom->mass = get_pte_mass(pteidx);
        atom->radius = get_pte_vdw_radius(pteidx);
      } else {
        badptecount++; /* unrecognized element */
      }
      strcpy(atom->type, atom->name);
      i++;
      break;
    default:
      break;
    }
  } while (rectype != PDB_END && rectype != PDB_EOF);

  fseek(pdb->fd, fpos, SEEK_SET);

  /* if all atoms are recognized, set the mass and radius flags too,  */
  /* otherwise let VMD guess these for itself using it's own methods  */
  if (badptecount == 0) {
    *optflags |= MOLFILE_MASS | MOLFILE_RADIUS;
  }

  return MOLFILE_SUCCESS;
}

static int read_next_timestep(void *v, int natoms, molfile_timestep_t *ts) {
  pdbdata *pdb = (pdbdata *)v;
  char pdbstr[PDB_BUFFER_LENGTH];
  int indx, i;
  float *x, *y, *z;
  float occup[1], beta[1];
  if (ts) {
    x = ts->coords;
    y = x+1;
    z = x+2;
  } else {
    x = y = z = 0;
  }
  i = 0;
  if (!pdb->fd) 
    return MOLFILE_ERROR;

  /* Read the rest of the frames in the current fd.  If there aren't any
   * more close it and go on to the next one.  If there aren't any more frames,
   * return MOLFILE_ERROR (-1);
   */
  
  while (i < pdb->natoms) {
    indx = read_pdb_record(pdb->fd, pdbstr);
    if(indx == PDB_ATOM) {
      /* just get the coordinates, and store them */
      if (ts) {
        get_pdb_coordinates(pdbstr, x, y, z, occup, beta);
        x += 3;
        y += 3;
        z += 3;
        i++;
      }
    } else if (indx == PDB_CRYST1) {
      if (ts) {
        get_pdb_cryst1(pdbstr, &ts->alpha, &ts->beta, &ts->gamma,
                               &ts->A, &ts->B, &ts->C);
      }
    } else if (indx == PDB_EOF) {
      if (i == 0) {
        /* Need to start a new frame, if possible */
        fclose(pdb->fd);
        pdb->fd = 0;
        vmd_delete_file(pdb->current_file);
        free(pdb->current_file);
        pdb->current_file = 0;
        pdb->babel_i++;
        if (pdb->babel_i >= pdb->babel_num) 
          return MOLFILE_ERROR; 
        pdb->current_file = file(pdb->original_file, pdb->babel_i, pdb->babel_num > 1); 
        pdb->fd = fopen(pdb->current_file, "r");
        if (!pdb->fd) {
          fprintf(stderr, 
            "Couldn't read babel output file %s\n", pdb->current_file); 
          free(pdb->current_file);
          pdb->current_file = 0;
          return MOLFILE_ERROR; 
        } 
      } else {
        /* premature end */
        fprintf(stderr, "PDB file %s contained too few atoms\n", pdb->current_file);
        return MOLFILE_ERROR;
      }
    }
  }

  return MOLFILE_SUCCESS; 
}

/* 
 * Free the pdb handle, and delete all the babel temp files.
 */
static void close_pdb_read(void *v) {
  pdbdata *pdb = (pdbdata *)v;
  if (!pdb) return;
  if (pdb->fd) {
    fclose(pdb->fd);
    pdb->fd = 0;
    vmd_delete_file(pdb->current_file);
    free(pdb->current_file);
  }
  free(pdb);
}



/*
 * Initialization stuff down here
 */

static molfile_plugin_t *plugins;
static int nplugins;

VMDPLUGIN_API int VMDPLUGIN_init() {
#if defined(_MSC_VER)
  return VMDPLUGIN_SUCCESS;
#else
  /* register all Babel 1.6 conversion options */
  const char **s = babel16filetypenames;
  int i;
  nplugins = 0;
  while (*s) { nplugins++; s++; }
  plugins = (molfile_plugin_t*)calloc(nplugins, sizeof(molfile_plugin_t));
  for (i=0; i<nplugins; i++) {
    plugins[i].abiversion = vmdplugin_ABIVERSION;         /* ABI version */
    plugins[i].type = MOLFILE_CONVERTER_PLUGIN_TYPE;      /* type of plugin */
    plugins[i].name = babel16filetypenames[i];            /* name of plugin */
    plugins[i].prettyname = babel16filetypenames[i];      /* name of plugin */
    plugins[i].author = "Justin Gullingsrud, John Stone"; /* author */
    plugins[i].majorv = 1;                                /* major version */
    plugins[i].minorv = 12;                               /* minor version */
    plugins[i].is_reentrant = VMDPLUGIN_THREADUNSAFE;     /* is not reentrant */
    plugins[i].filename_extension = babel16filetypes[i];  /* file extension */
    plugins[i].open_file_read = open_babel_read;
    plugins[i].read_structure = read_pdb_structure;
    plugins[i].read_next_timestep = read_next_timestep;
    plugins[i].close_file_read = close_pdb_read;
  }

#if 0
  /* register all Open Babel 1.100.2 conversion options */
  const char **s = openbabel11filetypenames;
  int i;
  nplugins = 0;
  while (*s) { nplugins++; s++; }
  plugins = (molfile_plugin_t*)calloc(nplugins, sizeof(molfile_plugin_t));
  for (i=0; i<nplugins; i++) {
    plugins[i].abiversion = vmdplugin_ABIVERSION;         /* ABI version */
    plugins[i].type = MOLFILE_CONVERTER_PLUGIN_TYPE;      /* type of plugin */
    plugins[i].shortname = openbabel11filetypenames[i];   /* name of plugin */
    plugins[i].prettyname = openbabel11filetypenames[i];  /* name of plugin */
    plugins[i].author = "Justin Gullingsrud, John Stone"; /* author */
    plugins[i].majorv = 2;                                /* major version */
    plugins[i].minorv = 11;                               /* minor version */
    plugins[i].is_reentrant = VMDPLUGIN_THREADUNSAFE;     /* is not reentrant */
    plugins[i].filename_extension = openbabel11filetypes[i];  /* file extension */
    plugins[i].open_file_read = open_babel_read;
    plugins[i].read_structure = read_pdb_structure;
    plugins[i].read_next_timestep = read_next_timestep;
    plugins[i].close_file_read = close_pdb_read;
  }
#endif

  return VMDPLUGIN_SUCCESS;
#endif
}

VMDPLUGIN_API int VMDPLUGIN_register(void *v, vmdplugin_register_cb cb) {
#if defined(_MSC_VER)
  return VMDPLUGIN_SUCCESS;
#else
  int i;
  for (i=0; i<nplugins; i++) {
    (*cb)(v, (vmdplugin_t *)(plugins+i));
  }
  return VMDPLUGIN_SUCCESS;
#endif
}

VMDPLUGIN_API int VMDPLUGIN_fini() {
#if defined(_MSC_VER)
  return VMDPLUGIN_SUCCESS;
#else
  free(plugins);
  nplugins = 0;
  plugins = 0;
  return VMDPLUGIN_SUCCESS;
#endif
}


#ifdef TEST_BABEL_PLUGIN

int main(int argc, char *argv[]) {
  molfile_header_t header;
  molfile_timestep_t timestep;
  void *v;

  while (--argc) {
    ++argv;
    v = open_babel_read(*argv, "xyz", &header);
    if (!v) {
      fprintf(stderr, "open_babel_read failed for file %s\n", *argv);
      return 1;
    }
    timestep.coords = (float *)malloc(3*sizeof(float)*header.numatoms);
    while (!read_next_timestep(v, &timestep));
    close_pdb_read(v);
  }
  return 0;
}


#endif


