/*
 *  Amira File Format (.am)
 *
 *  Supports scalar and vector fields, and a variety of triangle meshes.
 *  
 *  File format information and parsing approaches for reference and 
 *  comparison with our own:
 *    http://docs.drabba.net/usersguide/amira/5.3.3/amiramesh/HxFileFormat_AmiraMesh.html
 *    https://people.mpi-inf.mpg.de/~weinkauf/notes/amiramesh.html
 *    http://neuronland.org/NLMorphologyConverter/MorphologyFormats/AmiraMesh/Spec.html
 *    https://www.mathworks.com/matlabcentral/fileexchange/34909-loadamiramesh?requestedDomain=www.mathworks.com
 *    https://www.mathworks.com/matlabcentral/fileexchange/34909-loadamiramesh/content/loadAmiraMesh.m
 *    https://github.com/dune-project/dune-grid/blob/master/dune/grid/io/file/amiramesh/amirameshreader.cc
 *    https://parcomp-git.iwr.uni-heidelberg.de/marian/dune-common/commit/c184290ea19b14358c053441e3a55f6639d4f6c4.patch
 *    https://gitlab.dune-project.org/dominic/dune-grid/blob/6edbd89b8be2778d3940d1d293ce0c53b814cf0b/dune/grid/io/file/amiramesh/amirameshreader.cc
 *    https://github.com/openmicroscopy/bioformats/blob/v5.3.0-m3/components/formats-gpl/src/loci/formats/in/AmiraReader.java
 */

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

#if defined(_AIX)
#include <strings.h>
#endif

#if defined(WIN32) || defined(WIN64)
#define strcasecmp stricmp
#endif

#include "molfile_plugin.h"

// internal buffer size
#define BUFSZ 1024


typedef struct {
  FILE *f;
  char *headerbuf;
  int isbinary;
  int islittleendian;
  long dataoffset;

  // triangle mesh data
  int hastrimesh;
  int nvertices;
  int ntriangles;

  // volumetric  data
  int nsets;
  molfile_volumetric_t *vol;
} am_t;


static char *get_next_noncomment_line(char *buf, int bufsz, FILE *f) {
  while (1) {
    char *res = fgets(buf, bufsz, f);
    if (!res || (res[0] != '#' && res[0] != '\n' && res[0] != '\r'))
      return res;
  }
}

static long find_data_offset(FILE *f) {
  char buf[BUFSZ];
  while (1) {
    char *res = fgets(buf, BUFSZ, f);
    if (!res || (res[0] != '#' && res[0] != '\n' && res[0] != '\r')) {
      if (!res) {
        return -1;
      } else if (strstr(res, "# Data section follows")) {
        return ftell(f);
        // XXX fseek()/ftell() are incompatible with 64-bit LFS I/O 
        // implementations, hope we don't read any files >= 2GB...
      }
    }
  }
}


static int amira_readvar_int(am_t *am, char *varname, int *val) {
  char *pos = strstr(am->headerbuf, varname);
  if (pos != NULL) {
    if (sscanf(am->headerbuf, "%*s %d", val) == 2) {
      return 0;
    }
    printf("amiraplugin) failed to find variable '%s' in header\n", varname);
    return -1;
  }

  return -1;
}


static int amira_readvar_float(am_t *am, char *varname, float *val) {
  char *pos = strstr(am->headerbuf, varname);
  if (pos != NULL) {
    if (sscanf(am->headerbuf, "%*s %f", val) == 2) {
      return 0;
    }
    printf("amiraplugin) failed to find variable '%s' in header\n", varname);
    return -1;
  }

  return -1;
}


static int amira_check_trimesh(am_t *am) {
  // see if this file contains a triangle mesh or not
  if ((strstr(am->headerbuf, "Vertices { float[3] Vertices }") != NULL) &&
      (strstr(am->headerbuf, "TriangleData { int[7] Triangles }") != NULL)) {
    int rc;
    rc=amira_readvar_int(am, "nVertices", &am->nvertices);
    if (!rc) rc=amira_readvar_int(am, "nVertices", &am->ntriangles);
    if (!rc) {
      am->hastrimesh = 1;
      printf("amiraplugin) Found RGBA triangle mesh: %d verts, %d tris\n",
             am->nvertices, am->ntriangles);
      return 1;
    } else {
      printf("amiraplugin) Failed to find vertex/triangle counts for mesh!\n");
    }
  }
  return 0;
}


static void *open_file_read(const char *filepath, const char *filetype,
                            int *natoms) {
  FILE *f=NULL;
  am_t *am=NULL;
  *natoms = 0;
  
  f = fopen(filepath, "rb");
  if (!f) {
    printf("amiraplugin) Error opening file.\n");
    return NULL;
  }

  // read file header 
  char linebuf[BUFSZ];
  fgets(linebuf, BUFSZ, f);
  if (strncmp(linebuf, "# AmiraMesh", strlen("# AmiraMesh")) != 0) {
    printf("amiraplugin) Bad file header: '%s'\n", linebuf);
    return NULL;
  }

  char endianbuf[BUFSZ];
  char versionbuf[BUFSZ];
  char commentbuf[BUFSZ];
  int parsed=sscanf("# AmiraMesh %s %s %s", endianbuf, versionbuf, commentbuf);
  if (parsed < 2) {
    printf("amiraplugin) Bad file header: '%s'\n", linebuf);
    return NULL;
  }
 
  am = new am_t;
  memset(am, 0, sizeof(am_t));
  am->f = f;
  am->hastrimesh = 0;
  am->nsets = 0;
  am->vol = NULL;

  if (strstr(linebuf, "BINARY-LITTLE-ENDIAN") != NULL) {
    am->isbinary=1;
    am->islittleendian=1;
  } else if (strstr(linebuf, "BINARY-BIG-ENDIAN") != NULL) {
    am->isbinary=1;
    am->islittleendian=1;
  } else if (strstr(linebuf, "ASCII") != NULL) {
    am->isbinary=0;
    am->islittleendian=0;
  } else {
    printf("amiraplugin) Failed to parse header data format: '%s'\n",
           endianbuf);
    delete am;
    return NULL;
  }

  // A cheap strategy for avoiding difficult parsing for the few file variants
  // we really care about is to read all of the header into a giant string,
  // and then search for the variables and other bits we want using strstr()
  // or the like...
  am->dataoffset = find_data_offset(am->f);
  if (am->dataoffset < 0) {
    printf("amiraplugin) Failed to find data offset!\n");
    delete am;
    return NULL;
  }
  if (am->dataoffset > 1000000) {
    printf("amiraplugin) Header appears to be overly large, aborting read!\n");
    delete am;
    return NULL;
  }

  am->headerbuf = (char*) calloc(1, am->dataoffset); 
  rewind(am->f);
  fread(am->headerbuf, am->dataoffset, 1, am->f); 

  // check to see if the file contains a triangle mesh format
  // that we currently recognize.
  amira_check_trimesh(am);

  return am;
}


static void close_file_read(void *v) {
  am_t *am = (am_t *)v;

  fclose(am->f);
  if (am->vol != NULL)
    delete [] am->vol;
  delete am;
}


static int read_volumetric_metadata(void *v, int *nsets,
                                    molfile_volumetric_t **metadata) {
  am_t *am = (am_t *)v;
  *nsets = am->nsets;
  *metadata = am->vol;

  return MOLFILE_SUCCESS;
}


static int read_volumetric_data(void *v, int set, float *datablock,
                         float *colorblock) {
  return MOLFILE_ERROR;
//  return MOLFILE_SUCCESS;
}


static int read_rawgraphics(void *v, int *nelem, 
                            const molfile_graphics_t **data) {
#if 0
  int i, k, n;
  int nVert, nFaces, nEdges;
  float *vertices = NULL, *vertColors = NULL;
  char *vertHasColor = NULL;
  molfile_graphics_t *graphics = NULL;
  int j=0;

  char buff[BUFFLEN+1];
  FILE *infile = (FILE *)v;

  // First line is the header: "OFF"
  nextNoncommentLine(buff, BUFFLEN, infile);
  if (buff[0] != 'O' || buff[1] != 'F' || buff[2] != 'F') {
    fprintf(stderr, "offplugin) error: expected \"OFF\" header.\n");
    goto error;
  }

  // Second line: numVertices numFaces numEdges
  nextNoncommentLine(buff, BUFFLEN, infile);
  if (sscanf (buff, " %d %d %d", &nVert, &nFaces, &nEdges) < 2 || 
      nVert <= 0 || nFaces <= 0) {
    fprintf(stderr, "offplugin) error: wrong number of elements.\n");
    goto error;
  }

  // Read vertices
  vertices = (float *) calloc (3 * nVert, sizeof(float));
  vertHasColor = (char *) calloc (nVert, sizeof(char));
  vertColors = (float *) calloc (3 * nVert, sizeof(float));
  for (i = 0; i < nVert; i++) {
    nextNoncommentLine(buff, BUFFLEN, infile);
    int n = sscanf (buff, " %g %g %g %g %g %g", 
                    &vertices[3*i], &vertices[3*i+1], &vertices[3*i+2],
                    &vertColors[3*i], &vertColors[3*i+1], &vertColors[3*i+2]);
    if (n != 3 && n != 6) {
      fprintf(stderr, "offplugin) error: not enough data.\n");
      goto error;
    }
    vertHasColor[i] = (n == 6);
  }

  // Read faces
  // We alloc 6 times the memory because:
  //   -- a quadrangle will be transformed into two triangles.
  //   -- each triangle may have color, and then also its norm will be specified
  graphics = (molfile_graphics_t *) calloc(6*nFaces, sizeof(molfile_graphics_t));
  n = 0;
  for (i = 0; i < nFaces; i++) {
    int idx[4];
    float c[3];
    nextNoncommentLine(buff, BUFFLEN, infile);

    if (sscanf (buff, "%d", &k) != 1 || k < 3) {
      fprintf(stderr, "offplugin) error: not enough data.\n");
      goto error;
    }

    if (k > 4) {
      // TODO -- handle polygon decomposition into triangles
      // Follow the algorithm there:
      // http://www.flipcode.com/archives/Efficient_Polygon_Triangulation.shtml
      fprintf(stderr, "offplugin) error: TODO -- handling polygons with more than 4 vertices.\n");
      goto error;
    }

    if (k == 3) {
      j = sscanf (buff, "%d %d %d %d %g %g %g", &k, &idx[0], &idx[1], &idx[2], &c[0], &c[1], &c[2]);
      bool hasColor = ((j == 7) || (vertHasColor[idx[0]] && vertHasColor[idx[1]] && vertHasColor[idx[2]]));

      graphics[n].type = (hasColor ? MOLFILE_TRICOLOR : MOLFILE_TRIANGLE);
      graphics[n].data[0] = vertices[3*idx[0]  ];
      graphics[n].data[1] = vertices[3*idx[0]+1];
      graphics[n].data[2] = vertices[3*idx[0]+2];
      graphics[n].data[3] = vertices[3*idx[1]  ];
      graphics[n].data[4] = vertices[3*idx[1]+1];
      graphics[n].data[5] = vertices[3*idx[1]+2];
      graphics[n].data[6] = vertices[3*idx[2]  ];
      graphics[n].data[7] = vertices[3*idx[2]+1];
      graphics[n].data[8] = vertices[3*idx[2]+2];
      n++;

      if (j == 7) {
        // The facet has a specific color, use it.
        graphics[n].type = MOLFILE_NORMS;
        calcNormals (graphics[n-1].data, graphics[n].data);
        n++;

        graphics[n].type = MOLFILE_COLOR;
        graphics[n].data[0] = graphics[n].data[3] = graphics[n].data[6] = c[0];
        graphics[n].data[1] = graphics[n].data[4] = graphics[n].data[7] = c[1];
        graphics[n].data[2] = graphics[n].data[5] = graphics[n].data[8] = c[2];
        n++;
      } else if (hasColor) {
        // All three vertices have a color attribute
        graphics[n].type = MOLFILE_NORMS;
        calcNormals (graphics[n-1].data, graphics[n].data);
        n++;

        graphics[n].type = MOLFILE_COLOR;
        graphics[n].data[0] = vertColors[3*idx[0]  ];
        graphics[n].data[1] = vertColors[3*idx[0]+1];
        graphics[n].data[2] = vertColors[3*idx[0]+2];
        graphics[n].data[3] = vertColors[3*idx[1]  ];
        graphics[n].data[4] = vertColors[3*idx[1]+1];
        graphics[n].data[5] = vertColors[3*idx[1]+2];
        graphics[n].data[6] = vertColors[3*idx[2]  ];
        graphics[n].data[7] = vertColors[3*idx[2]+1];
        graphics[n].data[8] = vertColors[3*idx[2]+2];
        n++;
      }
    } else if (k == 4) {
      j = sscanf (buff, "%d %d %d %d %d %g %g %g", &k, &idx[0], &idx[1], &idx[2], &idx[3], &c[0], &c[1], &c[2]);
      bool hasColor = ((j == 8) || (vertHasColor[idx[0]] && vertHasColor[idx[1]] && vertHasColor[idx[2]] && vertHasColor[idx[3]]));

      // Split a quadrangle into two triangles
      graphics[n].type = (hasColor ? MOLFILE_TRICOLOR : MOLFILE_TRIANGLE);
      graphics[n].data[0] = vertices[3*idx[0]  ];
      graphics[n].data[1] = vertices[3*idx[0]+1];
      graphics[n].data[2] = vertices[3*idx[0]+2];
      graphics[n].data[3] = vertices[3*idx[1]  ];
      graphics[n].data[4] = vertices[3*idx[1]+1];
      graphics[n].data[5] = vertices[3*idx[1]+2];
      graphics[n].data[6] = vertices[3*idx[2]  ];
      graphics[n].data[7] = vertices[3*idx[2]+1];
      graphics[n].data[8] = vertices[3*idx[2]+2];
      n++;

      if (j == 8) {
        graphics[n].type = MOLFILE_NORMS;
        calcNormals (graphics[n-1].data, graphics[n].data);
        n++;
      
        graphics[n].type = MOLFILE_COLOR;
        graphics[n].data[0] = graphics[n].data[3] = graphics[n].data[6] = c[0];
        graphics[n].data[1] = graphics[n].data[4] = graphics[n].data[7] = c[1];
        graphics[n].data[2] = graphics[n].data[5] = graphics[n].data[8] = c[2];
        n++;
      } else if (hasColor) {
        graphics[n].type = MOLFILE_NORMS;
        calcNormals (graphics[n-1].data, graphics[n].data);
        n++;

        graphics[n].type = MOLFILE_COLOR;
        graphics[n].data[0] = vertColors[3*idx[0]];
        graphics[n].data[1] = vertColors[3*idx[0]+1];
        graphics[n].data[2] = vertColors[3*idx[0]+2];
        graphics[n].data[3] = vertColors[3*idx[1]];
        graphics[n].data[4] = vertColors[3*idx[1]+1];
        graphics[n].data[5] = vertColors[3*idx[1]+2];
        graphics[n].data[6] = vertColors[3*idx[2]];
        graphics[n].data[7] = vertColors[3*idx[2]+1];
        graphics[n].data[8] = vertColors[3*idx[2]+2];
        n++;
      }

      graphics[n].type = (hasColor ? MOLFILE_TRICOLOR : MOLFILE_TRIANGLE);
      graphics[n].data[0] = vertices[3*idx[2]];
      graphics[n].data[1] = vertices[3*idx[2]+1];
      graphics[n].data[2] = vertices[3*idx[2]+2];
      graphics[n].data[3] = vertices[3*idx[3]];
      graphics[n].data[4] = vertices[3*idx[3]+1];
      graphics[n].data[5] = vertices[3*idx[3]+2];
      graphics[n].data[6] = vertices[3*idx[0]];
      graphics[n].data[7] = vertices[3*idx[0]+1];
      graphics[n].data[8] = vertices[3*idx[0]+2];
      n++;

      if (j == 8) {
        graphics[n].type = MOLFILE_NORMS;
        calcNormals (graphics[n-1].data, graphics[n].data);
        n++;

        graphics[n].type = MOLFILE_COLOR;
        graphics[n].data[0] = graphics[n].data[3] = graphics[n].data[6] = c[0];
        graphics[n].data[1] = graphics[n].data[4] = graphics[n].data[7] = c[1];
        graphics[n].data[2] = graphics[n].data[5] = graphics[n].data[8] = c[2];
        n++;
      } else if (hasColor) {
        graphics[n].type = MOLFILE_NORMS;
        calcNormals (graphics[n-1].data, graphics[n].data);
        n++;

        graphics[n].type = MOLFILE_COLOR;
        graphics[n].data[0] = vertColors[3*idx[2]];
        graphics[n].data[1] = vertColors[3*idx[2]+1];
        graphics[n].data[2] = vertColors[3*idx[2]+2];
        graphics[n].data[3] = vertColors[3*idx[3]];
        graphics[n].data[4] = vertColors[3*idx[3]+1];
        graphics[n].data[5] = vertColors[3*idx[3]+2];
        graphics[n].data[6] = vertColors[3*idx[0]];
        graphics[n].data[7] = vertColors[3*idx[0]+1];
        graphics[n].data[8] = vertColors[3*idx[0]+2];
        n++;
      }
    }
  }

  *nelem = n;
  *data = (molfile_graphics_t *) realloc(graphics, n*sizeof(molfile_graphics_t));
  return MOLFILE_SUCCESS;

  // goto jump target for disaster handling: free memory and bail out
  error:
    free (graphics);
    free (vertices);
#endif

  printf("amiraplugin) read rawgraphics not implemented yet...\n");
  return MOLFILE_ERROR;
}




/*
 * Initialization stuff here
 */
static molfile_plugin_t plugin;

VMDPLUGIN_API int VMDPLUGIN_init(void) {
  memset(&plugin, 0, sizeof(molfile_plugin_t));
  plugin.abiversion = vmdplugin_ABIVERSION;
  plugin.type = MOLFILE_PLUGIN_TYPE;
  plugin.name = "off";
  plugin.prettyname = "Amira File Format (.am)";
  plugin.author = "John Stone";
  plugin.majorv = 0;
  plugin.minorv = 1;
  plugin.is_reentrant = VMDPLUGIN_THREADSAFE;
  plugin.filename_extension = "am";
  plugin.open_file_read = open_file_read;
  plugin.read_volumetric_metadata = read_volumetric_metadata;
  plugin.read_volumetric_data = read_volumetric_data;
  plugin.read_rawgraphics = read_rawgraphics;
  plugin.close_file_read = close_file_read;
  return VMDPLUGIN_SUCCESS;
}

VMDPLUGIN_API int VMDPLUGIN_register(void *v, vmdplugin_register_cb cb) {
  (*cb)(v, (vmdplugin_t *)&plugin);
  return VMDPLUGIN_SUCCESS;
}

VMDPLUGIN_API int VMDPLUGIN_fini(void) { return VMDPLUGIN_SUCCESS; }



