#include <string.h>
#include <stdlib.h>
#include "cionize_structs.h"
#include "cionize_grid.h"
#include "cionize_enermethods.h"

#include "cionize_userio.h"

int get_opts(cionize_params* params, cionize_api* molfiles,
    int argc, char** argv) {
  int i;
  char c;

  if (argc == 0) {
    print_usage(); 
    return 1;
  }

  if (argc == 1) {
    print_usage();
    return 1;
  }

  /* Parse the command line arguments */
  for (i=1; i<(argc-1); i++) {
    /* DH: why aren't we parsing with BSD getopt()? */
    /* DH: verify use of '-' to signal command line option */
    if (argv[i][0] != '-') {
      fprintf(stderr, "Error: Expecting command line option. Got %s\n",
          argv[i]);
      print_usage();
      return 1;
    }
    c = *(argv[i]+1);
    switch (c) {
      case 'p':
        i++;
        if (i >= argc) {
          fprintf(stderr, "Error: No argument for option `-%c'.\n", c);
          print_usage();
          return 1;
        }
        params->maxnumprocs = atoi(argv[i]);
        break;
      case 'i':
        i++;
        if (i >= argc) {
          fprintf(stderr, "Error: No argument for option `-%c'.\n", c);
          print_usage();
          return 1;
        }
        params->inputfile = argv[i];
        break;
      case 'f':
        i++;
        if (i >= argc) {
          fprintf(stderr, "Error: No argument for option '-%c'.\n", c);
          print_usage();
          return 1;
        }
        strncpy(molfiles->filetype, argv[i], 10);
        /* DH: need '\0' terminator after copying unknown string */
        molfiles->filetype[10] = '\0';
        break;
      case 'm':
        i++;
        if (i >= argc) {
          fprintf(stderr, "Error: No argument for option '-%c'.\n", c);
          print_usage();
          return 1;
        }
        if ((get_ener_method_from_string(argv[i], &(params->enermethod)))
            == -1) {
          params->enermethod = STANDARD;
          printf("Unrecognized method \"%s\", defaulting to STANDARD\n",
              argv[i]);
        }
        break;
#if defined(BINARY_GRIDFILE)
      case 'b':
        params->write_binary_gridfile = 1;
        break;
      case 'r':
        params->read_binary_gridfile = 1;
        break;
#endif
      default:
        fprintf (stderr, "Unknown option `-%c'.\n", c);
        print_usage();
        return 1;
    }
  }

  /*
  if (argc-optind != 2) {
    fprintf(stderr, "The input and output pdb file are required input!\n");
    print_usage();
    return 1;
  }
  */

  params->pdbin = argv[argc-1];

  return 0;
}

void print_usage() {
  printf( "\ncionize: Place ions by finding minima in a coulombic potential\n"
      "Usage: cionize (-p numprocs) (-i configfile) "
      "(-m calculation_type) (-f inputformat) input\n"
      "Type help for instructions once the program is running\n\n");
}

/* Open the input file for reading */
int open_input(cionize_params* params) {
  if (params->inputfile == NULL || strlen(params->inputfile) == 0) {
    printf("Warning: You didn't specify an input file. Using STDIN...\n");
    params->incmd = stdin;
  } else if (strncmp(params->inputfile, "-", 1) == 0) {
    params->incmd = stdin;
  } else {
    params->incmd = fopen(params->inputfile, "r");
    if (params->incmd == NULL) {
      fprintf(stderr, "Error: Couldn't open input file %s!!!\n"
          "Exiting...\n", params->inputfile);
      return 1;
    }
  }
  
  return 0;
}

/* Go through our initial input loop, set all the parameters, and get ready
 * for the run */
int settings_inputloop(cionize_params* params, cionize_grid* grid) {

  char inbuf[80];
  char headbuf[80];
  float value;

  /* Big loop to go through until people are done with setup */
  while ((strncasecmp(headbuf, "BEGIN", 5) != 0) && printf("\n>>> ") &&
      fgets(inbuf, 80, params->incmd)) {
    /* DH: you do know there are better ways to parse input, right? */
    if (sscanf(inbuf, "%s %f", headbuf, &value) != 2
        && strncasecmp(headbuf, "GRIDFILE", 8) != 0
        && strncasecmp(headbuf, "BEGIN", 5) != 0
        && strncasecmp(headbuf, "MULTIGRID", 9) != 0
        && strncasecmp(headbuf, "HELP", 4) != 0
        && strncasecmp(headbuf, "SHOWOPT", 7) != 0
#if defined(BINARY_GRIDFILE)
        && strncasecmp(headbuf, "BINWRITE", 8) != 0
        && strncasecmp(headbuf, "BINREAD", 7) != 0
#endif
        ) {
      printf("Error: Couldn't parse input line\n");
      continue;
    }
    
    if (strncasecmp(headbuf, "R_ION_SOLUTE",12)==0) {
      params->r_ion_prot = value;
    } else if (strncasecmp(headbuf, "R_ION_ION",9)==0) {
      params->r_ion_ion = value;
    } else if (strncasecmp(headbuf, "BORDERSIZE",10)==0) {
      params->bordersize = value;
    } else if (strncasecmp(headbuf, "GRIDSPACING", 11)==0) {
      grid->gridspacing = value;
    } else if (strncasecmp(headbuf, "GRIDFILE", 8)==0) {
      params->useoldgrid=1;
      sscanf(inbuf, "%*s %s", params->oldgridfile);
    } else if (strncasecmp(headbuf, "XYZDIM", 6)==0) {
      params->expsize = 1;
      sscanf(inbuf, "%*s %f %f %f %f %f %f",
          &(grid->minx), &(grid->miny), &(grid->minz),
          &(grid->maxx), &(grid->maxy), &(grid->maxz));
    } else if (strncasecmp(headbuf, "MULTIGRID", 9)==0) {
      params->enermethod &= MGMASK;
      params->enermethod |= MULTIGRID;
#if defined(BINARY_GRIDFILE)
    } else if (strncasecmp(headbuf, "BINWRITE", 8)==0) {
      params->write_binary_gridfile = 1;
    } else if (strncasecmp(headbuf, "BINREAD", 7)==0) {
      params->read_binary_gridfile = 1;
#endif
    } else if (strncasecmp(headbuf, "DDD", 3)==0) {
      if (params->enermethod != STANDARD) {
        printf("Warning: Distance dependent dielectric not yet supported "
            "for this type of calculation. Ignoring...\n");
        continue;
      }
      if (value <= 0) {
        printf("Warning: Ignoring dielectric constant <= 0\n");
        continue;
      }
      params->ddd = value;
      params->enermethod = DDD;
    } else if (strncasecmp(headbuf, "HELP", 4)==0) {
        printf("Recognized input:\n"
            "\tR_ION_SOLUTE--Minimum ion-solute distance\n"
            "\tR_ION_ION--Minimum ion-ion distance\n"
            "\tBORDERSIZE--Distance beyond solute to extend calculations\n"
            "\tGRIDSPACING--Grid density in angstroms\n"
            "\tDDD--Distance dependent dielectric constant "
            "(in inverse angstroms)\n"
            "\tGRIDFILE--Use energy grid from a file instead of "
            "calculating a new one\n"
            "\tXYZDIM--X, Y, and Z minima and maxima to use in calculations\n"
            "\tSHOWOPT--Show the values of current settings\n"
            "\tBEGIN--Calculate the energy grid with the "
            "existing parameters\n");
    } else if (strncasecmp(headbuf, "SHOWOPT", 7)==0) {
      printf( "\nCurrent options:\n"
          "\tIon-solute distance: %f\n"
          "\tIon-Ion distance: %f\n"
          "\tGrid spacing: %f\n"
          "\tBoundary size: %f\n"
          "\tMax. Processors: %i\n",
          params->r_ion_prot, params->r_ion_ion,
          grid->gridspacing, params->bordersize, params->maxnumprocs);
      if (params->ddd != 0) {
        printf("\tDistance dependent dielectric: %f * r\n", params->ddd);
      }
    } else if (strncasecmp(headbuf, "BEGIN", 5)!=0){
      fprintf(stderr, "Error: Unrecognized line in input file. "
          "Use HELP for options.\n");
    } else {
      break;
    }

  }

  if (strncasecmp(headbuf, "BEGIN", 5) != 0) {
    fprintf(stderr, "Warning: No BEGIN statement encountered. Exiting...\n");
    return 1;
  }

  return 0;
}

int get_ener_method_from_string(const char *methodstring, int *emethod) {
  int stat = 0;  /* status */
  /* Set the proper energy method, or leave it as normal
   * See cionize_enermethods.h for details */
  if (strcasecmp(methodstring, "standard")==0) {
    *emethod = STANDARD;
  }
  else if (strcasecmp(methodstring, "double")==0) {
    *emethod = DOUBLEPREC;
  }
  else if (strcasecmp(methodstring, "ddd")==0) {
    *emethod = DDD;
  }
  else if (strcasecmp(methodstring, "multigrid")==0) {
    *emethod &= MGMASK;        /* reset anything not MSM */
    *emethod |= MULTIGRID;     /* set MSM flag */
  }
  else if (strcasecmp(methodstring, "mlatcut01")==0) {
    *emethod &= MGMASK;        /* reset anything not MSM */
    *emethod &= ~MLATCUTMASK;  /* reset previous MLATCUTxx */
    *emethod |= (MLATCUT01 | MULTIGRID);
  }
  else if (strcasecmp(methodstring, "mlatcut02")==0) {
    *emethod &= MGMASK;        /* reset anything not MSM */
    *emethod &= ~MLATCUTMASK;  /* reset previous MLATCUTxx */
    *emethod |= (MLATCUT02 | MULTIGRID);
  }
  else if (strcasecmp(methodstring, "mlatcut03")==0) {
    *emethod &= MGMASK;        /* reset anything not MSM */
    *emethod &= ~MLATCUTMASK;  /* reset previous MLATCUTxx */
    *emethod |= (MLATCUT03 | MULTIGRID);
  }
  else if (strcasecmp(methodstring, "mlatcut04")==0) {
    *emethod &= MGMASK;        /* reset anything not MSM */
    *emethod &= ~MLATCUTMASK;  /* reset previous MLATCUTxx */
    *emethod |= (MLATCUT04 | MULTIGRID);
  }
  else if (strcasecmp(methodstring, "mbinlarge")==0) {
    *emethod &= MGMASK;        /* reset anything not MSM */
    *emethod &= ~MBINMASK;     /* reset previous MBINxx */
    *emethod |= (MBINLARGE | MULTIGRID);
  }
  else if (strcasecmp(methodstring, "mbinsmall")==0) {
    *emethod &= MGMASK;        /* reset anything not MSM */
    *emethod &= ~MBINMASK;     /* reset previous MBINxx */
    *emethod |= (MBINSMALL | MULTIGRID);
  }
  else if (strcasecmp(methodstring, "dev0")==0) {
    *emethod &= MGMASK;        /* reset anything not MSM */
    *emethod &= ~MDEVMASK;     /* reset previous MDEVxx */
    *emethod |= (MDEV0 | MULTIGRID);
  }
  else if (strcasecmp(methodstring, "dev1")==0) {
    *emethod &= MGMASK;        /* reset anything not MSM */
    *emethod &= ~MDEVMASK;     /* reset previous MDEVxx */
    *emethod |= (MDEV1 | MULTIGRID);
  }
  else if (strcasecmp(methodstring, "dev2")==0) {
    *emethod &= MGMASK;        /* reset anything not MSM */
    *emethod &= ~MDEVMASK;     /* reset previous MDEVxx */
    *emethod |= (MDEV2 | MULTIGRID);
  }
  else if (strcasecmp(methodstring, "dev3")==0) {
    *emethod &= MGMASK;        /* reset anything not MSM */
    *emethod &= ~MDEVMASK;     /* reset previous MDEVxx */
    *emethod |= (MDEV3 | MULTIGRID);
  }
  else {
    stat = -1;  /* unrecognized method */
  }
  return stat;
}


