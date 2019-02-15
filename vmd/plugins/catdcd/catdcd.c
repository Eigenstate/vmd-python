
/*****************************************************************************/
/*                                                                           */
/* (C) Copyright 2001-2005 Justin Gullingsrud and the University of Illinois.*/
/*                                                                           */
/*****************************************************************************/
/* $Id: catdcd.c,v 1.7 2017/08/30 16:18:57 johns Exp $                                                                      */
/*****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* 
 * Plugin header files; get plugin source from www.ks.uiuc.edu/Research/vmd"
 */
#include "libmolfile_plugin.h"
#include "molfile_plugin.h"
#include "hash.h"

#define CATDCD_MAJOR_VERSION 5
#define CATDCD_MINOR_VERSION 2


/* Set the maximum direct I/O aligned block size we are willing to support */
#define TS_MAX_BLOCKIO 4096

/* allocate memory and return a pointer that is aligned on a given   */
/* byte boundary, to be used for page- or sector-aligned I/O buffers */
/* We use this since posix_memalign() is not widely available...     */
#if 1
/* sizeof(unsigned long) == sizeof(void*) */
#define myintptrtype unsigned long
#elif 1
/* sizeof(size_t) == sizeof(void*) */
#define myintptrtype size_t
#else
/* C99 */
#define myintptrtype uintptr_t
#endif

static void *alloc_aligned_ptr(size_t sz, size_t blocksz, void **unalignedptr) {
  // pad the allocation to an even multiple of the block size
  size_t padsz = (sz + (blocksz - 1)) & (~(blocksz - 1));
  void * ptr = malloc(padsz + blocksz + blocksz);
  *unalignedptr = ptr;
  return (void *) ((((myintptrtype) ptr) + (blocksz-1)) & (~(blocksz-1)));
}


/*
 * Read indices specifying which atoms to keep.  Return number of indices,
 * or -1 on error.
 */

static void usage(const char *s);

static int dcdindices(const char *fname, int **indices) {
  FILE *fd;
  int *ind;
  int num, max;

  max = 10;
  ind = (int *)malloc(max*sizeof(int));
  num = 0;
  fd = fopen(fname, "r");
  if (fd == NULL) {
    fprintf(stderr, "Error opening index file %s\n", fname);
    return -1;
  }
  while (fscanf(fd, "%d", ind+num) == 1) {
    num++;
    if (num == max) {
      max *= 2;
      ind = (int *)realloc(ind, max*sizeof(int));
    }
  }
  fclose(fd);
  *indices = ind;
  return num;
}

typedef struct {
  int first;
  int last;
  int stride;
  const char *outfile;
  const char *otype;
  const char *sfile;   /* structure file */
  const char *stype;   /* structure file type */
  int optflags;        /* optflags from read_structure */
  molfile_atom_t *atoms;
  int num_ind;
  int *ind;
} catdcd_opt_t;

/*
 * Parse args, putting results in opt.  Return the new argc,  
 * or -1 on error.
 */
static int parse_args(int argc, char *argv[], catdcd_opt_t *opt) {
  if (argc < 2) usage(argv[0]);
  argc--;
  argv++;
  while (argc > 1) {
    if (!strcmp(argv[0], "-i")) {
      printf("Reading indices from file '%s'\n", argv[1]);
      opt->num_ind = dcdindices(argv[1], &opt->ind);
      if (opt->num_ind < 0) {
        fprintf(stderr, "Error reading index file.\n");
        return -1;
      }
      if (opt->num_ind < 1) {
        fprintf(stderr, "Error: no indices found in file.\n");
        return -1;
      }
      argc -= 2;
      argv += 2;
      continue;
    } else if (!strcmp(argv[0], "-first")) {
      opt->first = atoi(argv[1]);
      argc -= 2;
      argv += 2;
      continue;
    } else if (!strcmp(argv[0], "-last")) {
      opt->last = atoi(argv[1]);
      argc -= 2;
      argv += 2;
      continue;
    } else if (!strcmp(argv[0], "-stride")) {
      opt->stride = atoi(argv[1]);
      argc -= 2;
      argv += 2;
      continue;
    } else if (!strcmp(argv[0], "-o")) {
      opt->outfile = argv[1];
      argc -= 2;
      argv += 2;
      continue;
    } else if (!strcmp(argv[0], "-otype")) {
      opt->otype = argv[1];
      argc -= 2;
      argv += 2;
      continue;
    } else if (!strcmp(argv[0], "-s")) {
      opt->sfile = argv[1];
      argc -= 2;
      argv += 2;
      continue;
    } else if (!strcmp(argv[0], "-stype")) {
      opt->stype = argv[1];
      argc -= 2;
      argv += 2;
      continue;
    } else if (!strcmp(argv[0], "-num")) {
      /* Silently accept for backward compatibility */
      argc -= 1;
      argv += 1;
      continue;
    }
    /* Unrecognized flag, so must be either a file type or an input file */
    break;
  }
  return argc;
}

#define MAX_PLUGINS 200
static hash_t pluginhash;
static int num_plugins=0;
static molfile_plugin_t *plugins[MAX_PLUGINS];

static molfile_plugin_t *get_plugin(const char *filetype) {
  int id;
  if ((id = hash_lookup(&pluginhash, filetype)) == HASH_FAIL) {
    fprintf(stderr, "No plugin found for filetype '%s'\n", filetype);
    return NULL;
  }
  return plugins[id];
}

/*
 * Examine the first arguments in argv to determine what sort of file to
 * load.  
 * Input:   argc, number of remaining arguments
 *          argv, the pointer to the set of command line arguments
 *          filetype: address of last used file type
 * Returns: Number of arguments consumed, or -1 on error.
 * Side effects: filetype points to new filetype (or doesn't change)
 *               filename points to new filename.
 */

int next_file(int argc, char **argv, const char **filetype, const char **filename) {
  if (argc < 1) return 0;
  if (argc == 1) {
    *filename = argv[0];
    return 1;
  }
  if (!strlen(argv[0])) {
    fprintf(stderr, "0-length argument encountered!\n");
    return -1;
  }
  if (argv[0][0] == '-') {
    /* must be file type */
    if (argc < 2) {
      /* No file, so just stop here. */
      return 0;
    }
    *filetype = argv[0]+1; /* Skip past the '-'. */
    *filename = argv[1];
    return 2;
  }
  *filename = argv[0];
  return 1;
}

static int register_cb(void *v, vmdplugin_t *p) {
  const char *key = p->name;
  if (num_plugins >= MAX_PLUGINS) {
    fprintf(stderr, "Exceeded maximum allowed number of plugins; recompile. :(\n");
    return VMDPLUGIN_ERROR;
  }
  if (hash_insert(&pluginhash, key, num_plugins) != HASH_FAIL) {
    fprintf(stderr, "Multiple plugins for file type '%s' found!", key); 
    return VMDPLUGIN_ERROR;
  }
  plugins[num_plugins++] = (molfile_plugin_t *)p;
  return VMDPLUGIN_SUCCESS;
}

static int map_atoms_to_inds(catdcd_opt_t *opts, int inatoms) {
  molfile_atom_t *newatoms;
  int i;

  newatoms = (molfile_atom_t *)calloc(opts->num_ind,sizeof(molfile_atom_t));
  for (i=0; i<opts->num_ind; i++) {
    int j = opts->ind[i];
    if (j < 0 || j >= inatoms) {
      fprintf(stderr, "Atom index #%d (%d) is out of range.\n", i, j);
      return 0;
    }
    memcpy(newatoms+i, opts->atoms+j, sizeof(molfile_atom_t));
  }
  free(opts->atoms);
  opts->atoms = newatoms;
  return 1;
}

static void usage(const char *s) {
  int i;
  printf("   %s -o outputfile [-otype <filetype>] [-i indexfile]\n"
         "      [-stype <filetype>] [-s structurefile]\n"
         "      [-first firstframe] [-last lastframe] [-stride stride]\n"
         "      [-<filetype>] inputfile1 [-<filetype>] inputfile2 ...\n",s);
  printf("\n\nAllowed input file types:\n");
  for (i=0; i<num_plugins; i++) {
    if (plugins[i]->read_next_timestep) 
      printf("%s ", plugins[i]->name);
  }
  printf("\n\nAllowed output file types:\n");
  for (i=0; i<num_plugins; i++) {
    if (plugins[i]->write_timestep) 
      printf("%s ", plugins[i]->name);
  }
  printf("\n");
  exit(1);
}


int main(int argc, char *argv[]) {

  catdcd_opt_t opts;
  void *h_in, *h_out;
  int inatoms, outatoms;
  molfile_timestep_t ts_in, ts_out;
  int rc, frames_read, frames_written, args_read;
  const char *filetype, *filename;
  molfile_plugin_t *outputapi, *api;
  void *in_ptr=NULL, *out_ptr=NULL;

  printf("CatDCD %d.%d\n", CATDCD_MAJOR_VERSION, CATDCD_MINOR_VERSION);
  opts.first = opts.stride = 1;
  opts.last = -1;
  opts.outfile = NULL;
  opts.otype = "dcd";
  opts.sfile = NULL;
  opts.stype = "pdb";
  opts.optflags = 0;
  opts.atoms = NULL;
  opts.num_ind = 0;
  opts.ind = NULL;

  hash_init(&pluginhash, 20);
  MOLFILE_INIT_ALL
  MOLFILE_REGISTER_ALL(NULL, register_cb)

  rc = parse_args(argc, argv, &opts);
  if (rc < 0) return 1;
  argv += argc-rc;
  argc = rc;

  if (opts.last != -1 && opts.last < opts.first) {
    fprintf(stderr, "Error: last must be greater than or equal to first.\n");
    return 1;
  }

  if (opts.outfile) {
    outputapi = get_plugin(opts.otype);
    if (!outputapi || !(outputapi->write_timestep)) {
      fprintf(stderr, "Cannot write timesteps in '%s' format.\n", opts.otype);
      return 1;
    }
    if (outputapi->write_structure && !opts.sfile && !getenv("VMDNOSTRUCT")) {
      fprintf(stderr, 
          "Cannot write timesteps in '%s' format without structure file.\n", 
          opts.otype);
      return 1;
    }
  } else {
    outputapi = NULL;
  }

  /*
   * Input file type defaults to dcd.
   */
  filetype = "dcd";
  filename = NULL;

  /*
   * If no structure file was given, peek at the header of the first input 
   * file to get the number of atoms.  Otherwise, use the number of atoms
   * in the struture file.
   * All input files must have this number of atoms. 
   */
  inatoms = 0;
  if (opts.sfile) {
    if (!(api = get_plugin(opts.stype)))
      return 1;
    h_in = api->open_file_read(opts.sfile, opts.stype, &inatoms);
    if (h_in && api->read_structure) {
      opts.atoms = (molfile_atom_t *)calloc(inatoms,sizeof(molfile_atom_t));
      api->read_structure(h_in, &opts.optflags, opts.atoms);
      if (opts.num_ind) {
        if (!map_atoms_to_inds(&opts, inatoms)) return 1;
      }
    }
  } else {
    if (!next_file(argc, argv, &filetype, &filename)) {
      fprintf(stderr, "No input files found!\n");
      return 1;
    }
    if (!(api = get_plugin(filetype)))
      return 1;
  
    h_in = api->open_file_read(filename, filetype, &inatoms);
  }
  if (!h_in) {
    fprintf(stderr, "Error: could not open file '%s' for reading.\n",
      filename);
    return 1;
  }
  api->close_file_read(h_in);
  if (inatoms < 1) {
    fprintf(stderr, "No atoms found in %s.\n", 
        opts.sfile ? "structure file" : "first input file");
    return 1;
  }
  outatoms = opts.num_ind ? opts.num_ind : inatoms;

  /*
   * Open the output file for writing, if there is one
   */
  if (opts.outfile) {
    printf("Opening file '%s' for writing.\n", opts.outfile);
    h_out = outputapi->open_file_write(opts.outfile, opts.otype, outatoms);
    if (!h_out) {
      fprintf(stderr, "Error: Unable to open output file '%s' for writing.\n",
          opts.outfile);
      return 1;
    }
    if (outputapi->write_structure && !getenv("VMDNOSTRUCT")) {
      if (opts.atoms) {
        outputapi->write_structure(h_out, opts.optflags, opts.atoms);
      } else {
        fprintf(stderr, "Output file format '%s' needs atom information in structure file.\n", opts.otype);
        return 1;
      }
    }
#if defined(TS_MAX_BLOCKIO)
    // If we supprot block-based direct I/O, we must use memory buffers
    // that are padded to a full block size, and
    ts_in.coords = (float *) alloc_aligned_ptr(3*inatoms * sizeof(float),
                                            TS_MAX_BLOCKIO, (void**) &in_ptr);
    if (opts.num_ind) {
      ts_out.coords = (float *) alloc_aligned_ptr(3*outatoms * sizeof(float),
                                            TS_MAX_BLOCKIO, (void**) &out_ptr);
    }
#else
    ts_in.coords = (float *)calloc(3*inatoms,sizeof(float));
    if (opts.num_ind) {
      ts_out.coords = (float *)calloc(3*outatoms,sizeof(float));
    }
#endif
#if vmdplugin_ABIVERSION > 10
    /* no support for velocities */
    ts_in.velocities = NULL;
    ts_out.velocities = NULL;
#endif

  } else {
    h_out = NULL;
  }


  frames_read = frames_written = 0;
  filetype = "dcd";
  while ((args_read = next_file(argc, argv, &filetype, &filename))) {
    int tmpatoms = 0;
    int frames_in_file = 0;
    int written_from_file = 0;
    if (!(api = get_plugin(filetype))) {
      return 1;
    }
    if (!(api->read_next_timestep)) {
      fprintf(stderr, "Cannot read timesteps from file of type '%s'\n",
          filetype);
    }

    h_in = api->open_file_read(filename, filetype, &tmpatoms);
    if (!h_in) {
      fprintf(stderr, "Error: could not open file '%s' for reading.\n",
          filename);
      return 1;
    }
    if (tmpatoms != inatoms) {
      fprintf(stderr, 
          "Error: %s file %s contains wrong number of atoms (%d)\n", 
          filetype, filename, tmpatoms);
      return 1;
    }
    printf("Opened file '%s' for reading.\n", filename);
    while (opts.last == -1 || frames_read < opts.last) {
      if (!opts.outfile || frames_read + 1 < opts.first 
          || ((frames_read + 1 - opts.first) % opts.stride)) {
        /* Skip this frame */
        rc = api->read_next_timestep(h_in, inatoms, NULL);
        if (rc == -1) break;
        if (rc < 0) {
          fprintf(stderr, "Error reading input file '%s' (error code %d, during skip of frame %d)\n", filename, rc, frames_read + 1);
          return 1;
        }
        frames_read++;
        frames_in_file++;
        continue;
      }
      rc = api->read_next_timestep(h_in, inatoms, &ts_in);
      if (rc == -1) break;
      if (rc < 0) {
        fprintf(stderr, "Error reading input file '%s' (error code %d, during read of frame %d)\n", filename, rc, frames_read + 1);
        return 1;
      }
      frames_read++;
      frames_in_file++;
      if (opts.num_ind) {
        int j;
        for (j=0; j<opts.num_ind; j++) {
          ts_out.coords[3*j  ] = ts_in.coords[3*opts.ind[j]  ];
          ts_out.coords[3*j+1] = ts_in.coords[3*opts.ind[j]+1];
          ts_out.coords[3*j+2] = ts_in.coords[3*opts.ind[j]+2];
        }
        ts_out.A = ts_in.A;
        ts_out.B = ts_in.B;
        ts_out.C = ts_in.C;
        ts_out.alpha = ts_in.alpha;
        ts_out.beta = ts_in.beta;
        ts_out.gamma = ts_in.gamma;
        rc = outputapi->write_timestep(h_out, &ts_out);
      } else {
        rc = outputapi->write_timestep(h_out, &ts_in);
      }
      if (rc) {
        fprintf(stderr, "Error writing coordinates frame.\n");
        return 1;
      }
      frames_written++;
      written_from_file++;
    }
    api->close_file_read(h_in);
    printf("Read %d frames from file %s", frames_in_file, filename);
    if (opts.outfile) {
      printf(", wrote %d.\n", written_from_file);
    } else {
      printf(".\n");
    }
    argv += args_read;
    argc -= args_read;
  }
  printf("Total frames: %d\n", frames_read);
  if (opts.outfile) {
    outputapi->close_file_write(h_out);
#if defined(TS_MAX_BLOCKIO)
    free(in_ptr);
#else
    free(ts_in.coords);
#endif
    printf("Frames written: %d\n", frames_written);
  }
  if (opts.outfile && opts.num_ind) 
#if defined(TS_MAX_BLOCKIO)
    free(out_ptr);
#else
    free(ts_out.coords);
#endif
  if (opts.num_ind)
    free(opts.ind);

  printf("CatDCD exited normally.\n");
  return 0;
}

