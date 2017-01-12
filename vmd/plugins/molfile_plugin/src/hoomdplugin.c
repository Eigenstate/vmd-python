/***************************************************************************
 *
 * HOOMD and HOOMD-blue style XML format data/topology file reader and writer
 *
 * Copyright (c) 2009 Axel Kohlmeyer <akohlmey@gmail.com>
 *
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: hoomdplugin.c,v $
 *      $Author: akohlmey $       $Locker:  $             $State: Exp $
 *      $Revision: 1.16 $       $Date: 2011/12/06 04:57:12 $
 *
 ***************************************************************************/

/*
 * HOOMD-blue xml topology/trajectory file format:
 *
 * http://codeblue.umich.edu/hoomd-blue/doc/page_xml_file_format.html
 *
 * NOTES: 
 * - all XML elements are converted to lowercase before parsing, so case 
 *   will be ignored on reading. on writing only lowercase will be written.
 * - attributes are in general optional, but it is recommended to provide 
 *   them all. the plugin will print an informational message if it inserts 
 *   a missing attribute with a default value and a warning message if it 
 *   encounters a yet unknown attribute.
 * - units attributes are deprecated and always ignored.
 * - the angle, dihedral, improper nodes are only supported for 
 *   version 1.1 files. in files with no version attribute to hoomd_xml 
 *   or version 1.0 those nodes will be ignored.
 * - accelerations nodes appear in version 1.2. The molfile API
 *   has no method to pass them on, so they are currently ignored.
 * - charge nodes are only supported for version 1.3 and later files. 
 *   in files with no version attribute to hoomd_xml, version 1.0,
 *   or version 1.2, those nodes will be ignored.
 * - body nodes are supported starting with version 1.4. The body 
 *   value is mapped to the "resid" field in molfile.
 * - as of version 1.4 also the "vizsigma" attribute to the 
 *   configuration node is honored. It provides a scaling factor
 *   for the radius of particles and can be overridded by the 
 *   VMDHOOMDSIGMA environment variable. Both are ignored, however,
 *   if a diameter node is present.
 *
 ***************************************************

<?xml version ="1.0" encoding ="UTF-8" ?>
<hoomd_xml version="1.4">
  <!-- comments -->
  <configuration time_step="0" dimensions="3" vizsigma="0.5" ntypes="2">
    <box  lx="49.914" ly= "49.914" lz="49.914" />
    <position>
      3.943529 4.884501 12.317140
      3.985539 5.450660 13.107670
      3.521055 6.212838 13.516340
    </position>
    <type num="6">
      CT
      CM
      BB
      TRP
      LEU
      WS
    </type>
    <bond num="10">
      CT-CM 0 1
      CM-CT 1 2
      BB-TRP 3 4
      BB-BB  3 5
      BB-LEU 5 6
      BB-BB  5 7
      BB-TRP 7 8
      BB-BB  7 9
      BB-LEU 9 10
      BB-BB  9 3
    </bond>
    <angle num="9">
      CT-CM-CT   0  1  2
      BB-BB-TRP  9  3  4
      BB-BB-TRP  5  3  4
      BB-BB-LEU  3  5  6
      BB-BB-BB   3  5  7
      BB-BB-TRP  5  7  8
      BB-BB-BB   5  7  9
      BB-BB-LEU  7  9 10
      BB-BB-BB   7  9  3
    </angle>
    <diameter num="3">
      1.0 1.0 2.0
    </diameter>
    <mass num="3">
      1.0 1.0 2.0
    </mass>
    <charge num="3">
      -1.0 -1.0 2.0
    </charge>
    <body num="3">
      -1 0 0
    </charge>
    <wall>
      <coord ox="1.0" oy="2.0" oz="3.0" nx="4.0" ny="5.0" nz="6.0"/>
      <coord ox="7.0" oy="8.0" oz="9.0" nx="10.0" ny="11.0" nz="-12.0"/>
    </wall>
  </configuration>
</hoomd>

****************************************************/

#include "largefiles.h"   /* platform dependent 64-bit file I/O defines */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <errno.h>
#include "molfile_plugin.h"

#include "periodic_table.h"
#define THISPLUGIN plugin
#include "vmdconio.h"

/* XXX: it is a bit annoying to have to resolve to an external XML 
 * library for parsing. XML is messy and expat is plain c and makes
 * life much easier. perhaps a portable internal xml parser would 
 * do as well, but the one used in hoomd is in c++ and requires to 
 * the the whole file into memory as one gian string. :-( */
#include <expat.h>

/*! hoomd file format version that this plugin supports.
 *  it will refuse to load file formats with higher major
 *  version and print a warning on higher minor version.
 *  written files will be in exactly this version.
 */
#define HOOMD_FORMAT_MAJV 1
#define HOOMD_FORMAT_MINV 4

#define SMALL    1.0e-7f        /*! arbitrary small number. "almost zero"  */

/* numeric and symbolic representations of the supported xml nodes */
#define HOOMD_NONE      0       /**< undefined            */
#define HOOMD_XML       1       /**< toplevel tag         */
#define HOOMD_CONFIG    2       /**< timestep tag         */
#define HOOMD_BOX       3       /**< box dimensions tag   */
#define HOOMD_POS       4       /**< coordinates tag      */
#define HOOMD_IMAGE     5       /**< image counter tag    */
#define HOOMD_VEL       6       /**< velocities tag       */
#define HOOMD_TYPE      7       /**< atom type tag        */
#define HOOMD_BODY      8       /**< body number tag      */
#define HOOMD_MASS      9       /**< atom mass tag        */
#define HOOMD_CHARGE   10       /**< atom diameter tag    */
#define HOOMD_DIAMETER 11       /**< atom diameter tag    */
#define HOOMD_BOND     12       /**< bond section tag     */
#define HOOMD_ANGLE    13       /**< angle section tag    */
#define HOOMD_DIHEDRAL 14       /**< dihedral section tag */
#define HOOMD_IMPROPER 15       /**< improper section tag */
#define HOOMD_WALL     16       /**< wall description     */
#define HOOMD_COORD    17       /**< wall coordinate tag  */
#define HOOMD_ACCEL    18       /**< accelerations tag    */
#define HOOMD_ORIENT   19       /**< orientation tag      */
#define HOOMD_UNKNOWN  20       /**< unknown tag          */
#define HOOMD_NUMTAGS  21       /**< number of known tags */

/* This list has to stay consistend with the defines above */
static const char *hoomd_tag_name[HOOMD_NUMTAGS] = {
  "(none)", "hoomd_xml", "configuration", "box", /*  0 -  3 */
  "position", "image", "velocity",               /*  4 -  6 */
  "type", "body", "mass", "charge", "diameter",  /*  7 - 11 */
  "bond", "angle", "dihedral", "improper",       /* 12 - 15 */
  "wall", "coord", "acceleration", "orientation",/* 16 - 19 */
  "(error)"                                      /* 20 -    */
};

/*! maximum XML element nesting level supported */
#define HOOMD_MAXDEPTH 5

typedef struct {
  FILE *fp;
  XML_Parser p;
  void *buffer;
  int parsedepth;
  int parse_error;
  int currtag[HOOMD_MAXDEPTH+1];
  int counter;
  int majv, minv;
  int optflags;
  int numframe;
  int doneframe;
  int numdims;
  int numatoms;
  int numtypes;
  int numbonds;
  int numangles;
  int numdihedrals;
  int numimpropers;
  int numbondtypes;
  int numangletypes;
  int numdihedraltypes;
  int numimpropertypes;
  int *from;
  int *to;
  int *bondtype;
  int *angle;
  int *dihedral;
  int *improper;
  int *angletype;
  int *dihedraltype;
  int *impropertype;
  char **bondtypename;
  char **angletypename;
  char **dihedraltypename;
  char **impropertypename;
  char *filename;
  float mysigma;
  int *imagecounts;
  molfile_atom_t *atomlist;
#if vmdplugin_ABIVERSION > 10
  molfile_timestep_metadata_t ts_meta;
#endif
  molfile_timestep_t ts;
} hoomd_data_t;

/* forward declaration */
static molfile_plugin_t plugin;

/*
 * Error reporting macro for use in DEBUG mode
 */
#ifndef HOOMD_DEBUG
#define HOOMD_DEBUG 0
#endif
#if HOOMD_DEBUG
#define PRINTERR vmdcon_printf(VMDCON_ERROR,                            \
                               "\n In file %s, line %d: \n %s \n \n",   \
                               __FILE__, __LINE__, strerror(errno))
#else
#define PRINTERR (void)(0)
#endif

/* make sure pointers are NULLed after free(3)ing them. */
#define SAFE_FREE(ptr) free(ptr); ptr=NULL
/* calloc with test of success */
#define SAFE_CALLOC(ptr,type,count)       \
ptr = (type *)calloc(count,sizeof(type)); \
  if (ptr == NULL) {                      \
                                          \
    PRINTERR;                             \
    return MOLFILE_ERROR;                 \
  }

/*! convert string to upper case */
static void make_lower(char *s) 
{
  while (*s) {
    *s = tolower(*s);
    ++s;
  }
}

/*! find string in a NULL terminated list of strings.
 * grow storage and append if needed. return the index. */
static int addstring(char ***list, const char *key) 
{
  int i=0;
  char **p = *list;

  for (; p[i] != NULL;) {
    if (strcmp(p[i], key) == 0)
      return i;
    ++i;
  }
  
  *list        = realloc(*list, (i+2)*sizeof(char *));
  (*list)[i]   = strdup(key);
  (*list)[i+1] = NULL;
  
  return i;
}


/*! copy attribute/value pairs and convert to lowercase */
#define XML_ATTR_COPY(a,n,k,v)                          \
  strncpy(k, a[n],   sizeof(k)-1); make_lower(k);       \
  strncpy(v, a[n+1], sizeof(v)-1); make_lower(value)

/*! warning about unknown attribute */
#define XML_UNK_ATTR(e,k,v)                                         \
  vmdcon_printf(VMDCON_WARN, "hoomdplugin) ignoring unknown "       \
                " attribute %s='%s' for element %s\n", k, v, e);


/*! element handler for XML file format. 
 * this function is called whenever a new XML element is encountered.
 * this is a companion function to xml_end_tag(). We have to record
 * the nesting level of XML elements, so that we can we can return to
 * the previous element's state. hoomd doesn't use this property yet,
 * but may do so in the future.
 *
 * @param data  data structure with information on this file.
 * @param elmt  name of the XML element
 * @param attr  list of attributes and their values as strings.
 *              the order is attr1, val1, attr2, val2, ... 
 */
static void xml_new_tag(void *data, 
                        const XML_Char *elmt, 
                        const XML_Char **attr)
{
  int i, mytag;
  hoomd_data_t *d=data;
  char element[MOLFILE_BUFSIZ];
  char key[MOLFILE_BUFSIZ];
  char value[MOLFILE_BUFSIZ];
  

  if (d == NULL) return;

  d->parsedepth ++;

#if HOOMD_DEBUG
  for (i=0; i < d->parsedepth; ++i) {
    printf(">");
  }
  printf("%s",elmt);
  for (i=0;attr[i]; i+=2) {
    printf(" %s='%s'",attr[i],attr[i+1]);
  }
  printf("\n");
#endif

  strncpy(element, elmt, sizeof(element)-1);
  make_lower(element);

  /* convert text mode tags to numbers and parse attributes. */
  mytag = HOOMD_UNKNOWN;
  for (i=0; i < HOOMD_NUMTAGS; ++i) {
    if (strcmp(elmt, hoomd_tag_name[i]) == 0) {
      mytag = i;
      break;
    }
  }
  d->currtag[d->parsedepth] = mytag;
  d->counter = 0;
   
  switch (mytag) {

    case HOOMD_XML:             /* root node */
      for (i=0; attr[i]; i+=2) {
        XML_ATTR_COPY(attr, i, key, value);
        if(strcmp(key,"version") == 0) {
          d->majv = atoi(strtok(value, "."));
          d->minv = atoi(strtok(NULL,  "."));
        } else {
          XML_UNK_ATTR(element, key, value);
        }
      }
      if (d->majv < 0) {
        vmdcon_printf(VMDCON_INFO, "hoomdplugin) No version attribute found "
                      "on <%s> element assuming version=\"1.0\"\n", 
                      hoomd_tag_name[HOOMD_XML] );
        d->majv = 1;
        d->minv = 0;
      }
      break;

    case HOOMD_CONFIG:          /* new frame */
      for (i=0; attr[i]; i+=2) {
        XML_ATTR_COPY(attr, i, key, value);
        if(strcmp(key,"time_step") == 0) {
#if vmdplugin_ABIVERSION > 10
          d->ts.physical_time = atof(value);
#else 
          ;
#endif
        } else if(strcmp(key,"natoms") == 0) {
          d->numatoms = atoi(value);
        } else if(strcmp(key,"ntypes") == 0) {
          d->numtypes = atoi(value);
        } else if(strcmp(key,"dimensions") == 0) {
          if (d->majv == 1 && d->minv < 2) {
            vmdcon_printf(VMDCON_WARN, "hoomdplugin) Found dimensions "
                          "attribute in a pre-1.2 version format file. "
                          "Ignoring it...\n");
          } else {
            d->numdims = atoi(value);
          }
        } else if(strcmp(key,"vizsigma") == 0) {
          if (d->majv == 1 && d->minv < 4) {
            vmdcon_printf(VMDCON_WARN, "hoomdplugin) Found vizsigma "
                          "attribute in a pre-1.4 version format file. "
                          "Ignoring it...\n");
          } else {
            d->mysigma = atof(value);
          }
        } else {
          XML_UNK_ATTR(element, key, value);
        }
      }
      break;

    case HOOMD_BOX:             /* new box dimensions */
      /* hoomd only supports orthogonal boxes at the moment */
      d->ts.A     = d->ts.B    = d->ts.C     = 0.0f;
      d->ts.alpha = d->ts.beta = d->ts.gamma = 90.0f;
      for (i=0; attr[i]; i+=2) {
        XML_ATTR_COPY(attr, i, key, value);
        if (strcmp(key,"units") == 0) {
        } else if(strcmp(key,"lx") == 0) {
          d->ts.A = atof(value);
        } else if(strcmp(key,"ly") == 0) {
          d->ts.B = atof(value);
        } else if(strcmp(key,"lz") == 0) {
          d->ts.C = atof(value);
        } else {
          XML_UNK_ATTR(element, key, value);
        }
      }
      break;

      case HOOMD_POS:             /* coordinates */
      for (i=0; attr[i]; i+=2) {
        XML_ATTR_COPY(attr, i, key, value);
        if (strcmp(key,"units") == 0) {
          ;                     /* ignore */
        } else if (strcmp(key,"num") == 0) {
          if (d->numatoms < 1) {
            d->numatoms = atoi(value); /* number of positions. */
          }
        } else {
          XML_UNK_ATTR(element, key, value);
        }
      }
      break;

    case HOOMD_VEL:             /* velocities */
#if vmdplugin_ABIVERSION > 10
      d->ts_meta.has_velocities = 1;
      for (i=0; attr[i]; i+=2) {
        XML_ATTR_COPY(attr, i, key, value);
        if (strcmp(key, "units") == 0) {
          ;                     /* ignore */
        } else if (strcmp(key,"num") == 0) {
          ;                     /* XXX: number of velocities. use for check. */
        } else {
          XML_UNK_ATTR(element, key, value);
        }
      }
#endif
      break;

    case HOOMD_BODY:            /* body/resid tag */
      if (d->majv == 1 && d->minv < 4) {
        vmdcon_printf(VMDCON_WARN, "hoomdplugin) Found <%s> section in "
                      "a pre-1.4 version format file. Ignoring it...\n",
                      hoomd_tag_name[mytag]);
      } else {
        for (i=0; attr[i]; i+=2) {
          XML_ATTR_COPY(attr, i, key, value);
          if (strcmp(key,"num") == 0) {
            ;                     /* XXX: number of body tags. use for check. */
          } else {
            XML_UNK_ATTR(element, key, value);
          }
        }
      }
      break;

    case HOOMD_MASS:            /* atom mass */
      d->optflags |= MOLFILE_MASS;
      for (i=0; attr[i]; i+=2) {
        XML_ATTR_COPY(attr, i, key, value);
        if (strcmp(key,"units") == 0) {
          ;                     /* ignore */
        } else if (strcmp(key,"num") == 0) {
          ;                     /* XXX: number of masses. use for check. */
        } else {
          XML_UNK_ATTR(element, key, value);
        }
      }
      break;

    case HOOMD_CHARGE:          /* atom charge */
      if (d->majv == 1 && d->minv < 3) {
        vmdcon_printf(VMDCON_WARN, "hoomdplugin) Found charge "
                          "section in a pre-1.3 version format file. "
                          "Ignoring it...\n");
      } else {
        d->optflags |= MOLFILE_CHARGE;
        for (i=0; attr[i]; i+=2) {
          XML_ATTR_COPY(attr, i, key, value);
          if (strcmp(key,"units") == 0) {
            ;                     /* ignore */
          } else if (strcmp(key,"num") == 0) {
            ;                     /* XXX: number of masses. use for check. */
          } else {
            XML_UNK_ATTR(element, key, value);
          }
        }
      }
      break;

    case HOOMD_DIAMETER:          /* particle diameter */
      d->optflags |= MOLFILE_RADIUS;
      for (i=0; attr[i]; i+=2) {
        XML_ATTR_COPY(attr, i, key, value);
        if (strcmp(key,"units") == 0) {
          ;                     /* ignore */
        } else if (strcmp(key,"num") == 0) {
          ;                     /* XXX: number of diameters. use for check. */
        } else {
          XML_UNK_ATTR(element, key, value);
        }
      }
      break;

    case HOOMD_IMAGE:           /* fallthrough */
    case HOOMD_TYPE:            /* fallthrough */
    case HOOMD_BOND:
      for (i=0; attr[i]; i+=2) {
        XML_ATTR_COPY(attr, i, key, value);
        if (strcmp(key,"num") == 0) {
          ;                 /* XXX: number of atom types. use for check. */
        } else {
          XML_UNK_ATTR(element, key, value);
        }
      }
      break;

    /* angle type definitions */
    case HOOMD_ANGLE:           /* fallthrough */
    case HOOMD_DIHEDRAL:        /* fallthrough */
    case HOOMD_IMPROPER:
      if (d->majv == 1 && d->minv < 1) {
        if ( ((mytag == HOOMD_ANGLE) && (d->numangles < -1)) ||
             ((mytag == HOOMD_DIHEDRAL) && (d->numdihedrals < -1)) ||
             ((mytag == HOOMD_IMPROPER) && (d->numimpropers < -1)) ) {
          vmdcon_printf(VMDCON_WARN, "hoomdplugin) Found <%s> section in "
                        "a pre-1.1 version format file. Ignoring it...\n",
                        hoomd_tag_name[mytag]);
        } else {
          for (i=0; attr[i]; i+=2) {
            XML_ATTR_COPY(attr, i, key, value);
            if (strcmp(key,"num") == 0) {
              ;                 /* XXX: number of angles. use for check. */
            } else {
              XML_UNK_ATTR(element, key, value);
            }
          }
        }
      }
      break;

    case HOOMD_WALL:            /* wall section */
      for (i=0; attr[i]; i+=2) {
        XML_ATTR_COPY(attr, i, key, value);
        XML_UNK_ATTR(element, key, value);
      }
      break;

    case HOOMD_COORD:           /* wall coordinate */
      for (i=0; attr[i]; i+=2) {
        XML_ATTR_COPY(attr, i, key, value);
        if (strcmp(key,"units") == 0) {
          ;                       /* ignore */
        } else if(strcmp(key,"ox") == 0) {
          ; /* do nothing. we don't do anything with walls, yet.*/
        } else if(strcmp(key,"oy") == 0) {
          ; /* do nothing. we don't do anything with walls, yet.*/
        } else if(strcmp(key,"oz") == 0) {
          ; /* do nothing. we don't do anything with walls, yet.*/
        } else if(strcmp(key,"nx") == 0) {
          ; /* do nothing. we don't do anything with walls, yet.*/
        } else if(strcmp(key,"ny") == 0) {
          ; /* do nothing. we don't do anything with walls, yet.*/
        } else if(strcmp(key,"nz") == 0) {
          ; /* do nothing. we don't do anything with walls, yet.*/
        } else {
          XML_UNK_ATTR(element, key, value);
        }
      }
      break;

    case HOOMD_ACCEL:
      if (d->majv == 1 && d->minv < 2) {
        vmdcon_printf(VMDCON_WARN, "hoomdplugin) Found <%s> section in "
                        "a pre-1.2 version format file. Ignoring it...\n",
                      hoomd_tag_name[mytag]);
      } else {
        for (i=0; attr[i]; i+=2) {
          XML_ATTR_COPY(attr, i, key, value);
            if (strcmp(key,"num") == 0) {
              ;                 /* XXX: number of accelerations. use for check. */
            } else {
              XML_UNK_ATTR(element, key, value);
            }
          XML_UNK_ATTR(element, key, value);
        }
      }
      break;

    case HOOMD_ORIENT:            /* orientation tag */
      if (d->majv == 1 && d->minv < 4) {
        vmdcon_printf(VMDCON_WARN, "hoomdplugin) Found <%s> section in "
                      "a pre-1.4 version format file. Ignoring it...\n",
                      hoomd_tag_name[mytag]);
      } else {
        for (i=0; attr[i]; i+=2) {
          XML_ATTR_COPY(attr, i, key, value);
          if (strcmp(key,"num") == 0) {
            ;                     /* XXX: number of orientation tags. use for check. */
          } else {
            XML_UNK_ATTR(element, key, value);
          }
        }
      }
      break;

    default:
      d->currtag[d->parsedepth] = HOOMD_UNKNOWN;
      vmdcon_printf(VMDCON_WARN, "hoomdplugin) Ignoring unknown XML "
                    "element: %s.\n", element);
      break;
  }
}

/*! end of element handler for xml document. */
static void xml_end_tag(void *data, const XML_Char *elmt)
{
  hoomd_data_t *d=data;
  int mytag;

  if (d == NULL) return;

  mytag = d->currtag[d->parsedepth];
  switch (mytag) {

    case HOOMD_CONFIG:
      d->doneframe=1;
      break;

    case HOOMD_TYPE:
      /* store number of atoms the first time we read this section and
       * if it has not yet been passed on at the <configuration> tag.
       * VMD (currently) assumes that this doesn't change, so we
       * don't have to parse this more than once. */
      if (d->numatoms < 0) {    
        d->numatoms = d->counter;
      }
      break;

    case HOOMD_XML:             /* fallthrough. */
    case HOOMD_BOX:             /* fallthrough. */
    case HOOMD_IMAGE:           /* fallthrough. */
    case HOOMD_WALL:            /* fallthrough. */
    case HOOMD_COORD:           /* fallthrough. */
    case HOOMD_ACCEL:           /* fallthrough. */
    case HOOMD_ORIENT:          /* fallthrough. */
      ;                         /* nothing to do */
      break;

    case HOOMD_POS:
      /* assign the number of atoms if it has not been done before */
      if ( (d->counter > 0) && (d->numatoms < 0) ) {
        d->numatoms = d->counter/3;
      }                         /* fallthrough */
      
    case HOOMD_VEL:
      if ( (d->counter > 0) && (d->numatoms > 0) && (d->counter != (3*d->numatoms)) ) {
        vmdcon_printf(VMDCON_ERROR, "hoomdplugin) Inconsistent %s data. Expected %d, but "
                      "got %d items.\n", hoomd_tag_name[mytag], 3*d->numatoms, d->counter);
        d->parse_error = 1;
      }
      break;

    case HOOMD_MASS:            /* fallthrough. */
    case HOOMD_CHARGE:          /* fallthrough. */
    case HOOMD_BODY:            /* fallthrough. */
    case HOOMD_DIAMETER:
      if ( (d->counter > 0) && (d->numatoms > 0) && (d->counter != d->numatoms) )  {
        vmdcon_printf(VMDCON_ERROR, "hoomdplugin) Inconsistent %s data. Expected %d, but "
                      "got %d items.\n", hoomd_tag_name[mytag], d->numatoms, d->counter);
        d->parse_error = 1;
      }
      
      break;

    case HOOMD_BOND:
      /* store number of bonds the first time we read this section.
       * VMD (currently) assumes that this doesn't change, so we
       * don't have to do this more than once. */
      if (d->numbonds < 0) {
        d->numbonds = d->counter/3; /* three 'words' per bond */
      }
      break;

    case HOOMD_ANGLE:
      /* store number of angles the first time we read this section.
       * VMD (currently) assumes that this doesn't change, so we
       * don't have to do this more than once. */
      if (d->numangles < 0) {
        if (d->majv == 1 && d->minv > 0) { /* angle only valid from version 1.1 on */
          d->numangles = d->counter/4;    /* four 'words' per angle */
        } else {
          d->numangles = 0;
        }
      }
      break;

    case HOOMD_DIHEDRAL:
      /* store number of dihedrals the first time we read this section.
       * VMD (currently) assumes that this doesn't change, so we
       * don't have to do this more than once. */
      if (d->numdihedrals < 0) {
        if (d->majv == 1 && d->minv > 0) { /* dihedral only valid from version 1.1 on */
          d->numdihedrals = d->counter/5;    /* four 'words' per dihedral */
        } else {
          d->numdihedrals = 0;
        }
      }
      break;

    case HOOMD_IMPROPER:
      /* store number of impropers the first time we read this section.
       * VMD (currently) assumes that this doesn't change, so we
       * don't have to do this more than once. */
      if (d->numimpropers < 0) {
        if (d->majv == 1 && d->minv > 0) { /* improper only valid from version 1.1 on */
          d->numimpropers = d->counter/5;    /* four 'words' per improper */
        } else {
          d->numimpropers = 0;
        }
      }
      break;

    default: 
      if (mytag < HOOMD_NUMTAGS) {
        vmdcon_printf(VMDCON_WARN, "hoomdplugin) No end handler for HOOMD tag '%s'.\n",
                      hoomd_tag_name[mytag]);
      } else {
        vmdcon_printf(VMDCON_WARN, "hoomdplugin) Unknown HOOMD tag id: '%d'.\n", mytag);
      }
      
      break;
  }

  d->currtag[d->parsedepth] = HOOMD_NONE;
  d->parsedepth--;
  d->counter=0;
#if HOOMD_DEBUG
  printf("end of tag %s. parsedepth=%d\n",elmt,d->parsedepth);
#endif
}

/*! content data handler for xml document */
static void xml_data_block(void *data, const XML_Char *s, int len) 
{
  hoomd_data_t *d =NULL; 
  int i,lmax, mytag;

  d = (hoomd_data_t *)data;
  if (d == NULL) return;          /* XXX: bug in program */
  if (d->parsedepth < 1) return;  /* not yet within XML block */
  if (d->parsedepth > HOOMD_MAXDEPTH) return;  /* too much nesting */
  if (len < 1) return;

  mytag = d->currtag[d->parsedepth];
  switch (mytag) {
    
    case HOOMD_TYPE:

      /* count the number of atoms by counting the types */
      if (d->numatoms < 0) {
        char buffer[1024];
        char *p;
        lmax=1023;
        
        if (len < lmax) lmax=len;
        memcpy(buffer,s,lmax);
        buffer[lmax]='\0';

        p=strtok(buffer," \t\n");
        while (p) {
          d->counter ++;
          p=strtok(NULL," \t\n");
        }
      } else { 
        /* assign atom types the second time we parse the first frame */
        if (d->numframe < 2) {
          char buffer[1024];
          char *p;
          molfile_atom_t *atom;
          
          if (len==1 && ((*s == ' ') || (*s == '\n') || (*s == '\t'))) return;

          lmax=1023;
          if (len < lmax) lmax=len;
          memcpy(buffer,s,lmax);
          buffer[lmax]='\0';

          p=strtok(buffer," \t\n");
          while (p) {
            atom=d->atomlist + d->counter;
            if (atom == NULL) return;

            strncpy(atom->name, p, sizeof(atom->name));
            strncpy(atom->type, p, sizeof(atom->type));
            atom->atomicnumber = get_pte_idx(atom->type);
            d->counter ++;
            p=strtok(NULL," \t\n");
          }
        }
      }
      break;
      
    case HOOMD_BOND:
      /* count the number of bonds by counting the types */
      if ((d->numbonds < 0) || (d->numframe < 2)) {
        char buffer[1024];
        char *p;
        int idx, num, n;
        lmax=1023;
        
        if (len < lmax) lmax=len;
        memcpy(buffer,s,lmax);
        buffer[lmax]='\0';

        p=strtok(buffer," \t\n");
        while (p) {
          num = d->counter / 3;
          if (d->numbonds > 0) {
            i=d->counter % 3;
            if (i == 0) {
              n = d->numbondtypes;
              idx=addstring(&(d->bondtypename), p);
              if (idx < 0)
                d->parse_error = 1;
              if (idx >= n)
                d->numbondtypes=idx+1;
              d->bondtype[num] = idx;

              /* CAVEAT: atom indices start at 1 here. psf style. XXX */
            } else if (i == 1) { 
              d->from[num] = atoi(p) + 1;
            } else if (i == 2) {
              d->to[num] = atoi(p) + 1;
            }
          }
          d->counter ++;
          p=strtok(NULL," \t\n");
        }
      }
      break;
      
    case HOOMD_ANGLE:
      if (d->majv == 1 && d->minv > 0) {  /* angle is only valid from version 1.1 on */
        /* count the number of angles by counting the types */
        if ((d->numangles < 0) || (d->numframe < 2)) {
          char buffer[1024];
          char *p;
          int idx, num, n;
          lmax=1023;
        
          if (len < lmax) lmax=len;
          memcpy(buffer,s,lmax);
          buffer[lmax]='\0';

          p=strtok(buffer," \t\n");
          while (p) {
            num = d->counter / 4;
            if (d->numangles > 0) {
              i=d->counter % 4;
              if (i == 0) {
                n = d->numangletypes;
                idx=addstring(&(d->angletypename), p);
                if (idx < 0)
                  d->parse_error = 1;
                if (idx >= n)
                  d->numangletypes=idx+1;
                d->angletype[num] = idx;

                /* CAVEAT: atom indices have to start at 1 here. psf style. XXX */
              } else if (i == 1) { 
                d->angle[3*num  ] = atoi(p) + 1;
              } else if (i == 2) {
                d->angle[3*num+1] = atoi(p) + 1;
              } else if (i == 3) {
                d->angle[3*num+2] = atoi(p) + 1;
              }
            }
            d->counter ++;
            p=strtok(NULL," \t\n");
          }
        }
      }
      break;
      
    case HOOMD_DIHEDRAL:
      if (d->majv == 1 && d->minv > 0) {  /* dihedral is only valid from version 1.1 on */
        /* count the number of dihedrals by counting the types */
        if ((d->numdihedrals < 0) || (d->numframe < 2)) {
          char buffer[1024];
          char *p;
          int idx, num, n;
          lmax=1023;
        
          if (len < lmax) lmax=len;
          memcpy(buffer,s,lmax);
          buffer[lmax]='\0';

          p=strtok(buffer," \t\n");
          while (p) {
            num = d->counter / 5;
            if (d->numdihedrals > 0) {
              i=d->counter % 5;
              if (i == 0) {
                n = d->numdihedraltypes;
                idx=addstring(&(d->dihedraltypename), p);
                if (idx < 0)
                  d->parse_error = 1;
                if (idx >= n)
                  d->numdihedraltypes=idx+1;
                d->dihedraltype[num] = idx;

                /* CAVEAT: atom indices start at 1 here. psf style. XXX */
              } else if (i == 1) { 
                d->dihedral[4*num  ] = atoi(p) + 1;
              } else if (i == 2) {
                d->dihedral[4*num+1] = atoi(p) + 1;
              } else if (i == 3) {
                d->dihedral[4*num+2] = atoi(p) + 1;
              } else if (i == 4) {
                d->dihedral[4*num+3] = atoi(p) + 1;
              }
            }
            d->counter ++;
            p=strtok(NULL," \t\n");
          }
        }
      }
      break;
      
    case HOOMD_IMPROPER:
      if (d->majv == 1 && d->minv > 0) {  /* improper is only valid from version 1.1 on */
        /* count the number of impropers by counting the types */
        if ((d->numimpropers < 0) || (d->numframe < 2)) {
          char buffer[1024];
          char *p;
          int idx, num, n;
          lmax=1023;
        
          if (len < lmax) lmax=len;
          memcpy(buffer,s,lmax);
          buffer[lmax]='\0';

          p=strtok(buffer," \t\n");
          while (p) {
            num = d->counter / 5;
            if (d->numimpropers > 0) {
              i=d->counter % 5;
              if (i == 0) {
                n = d->numimpropertypes;
                idx=addstring(&(d->impropertypename), p);
                if (idx < 0)
                  d->parse_error = 1;
                if (idx >= n)
                  d->numimpropertypes=idx+1;
                d->impropertype[num] = idx;

                /* CAVEAT: atom indices start at 1 here. psf style. XXX */
              } else if (i == 1) { 
                d->improper[4*num  ] = atoi(p) + 1;
              } else if (i == 2) {
                d->improper[4*num+1] = atoi(p) + 1;
              } else if (i == 3) {
                d->improper[4*num+2] = atoi(p) + 1;
              } else if (i == 4) {
                d->improper[4*num+3] = atoi(p) + 1;
              }
            }
            d->counter ++;
            p=strtok(NULL," \t\n");
          }
        }
      }
      break;
      
    case HOOMD_DIAMETER:
      /* set radius, from diameter block. radius = 0.5*diameter. */
      if (d->numatoms > 0) {
        char buffer[1024];
        char *p;
        molfile_atom_t *atom;
          
        if (len==1 && ((*s == ' ') || (*s == '\n') || (*s == '\t'))) return;

        lmax=1023;
        if (len < lmax) lmax=len;
        memcpy(buffer,s,lmax);
        buffer[lmax]='\0';

        p=strtok(buffer," \t\n");
        while (p) {
          atom=d->atomlist + d->counter;
          if (atom == NULL) return;

          atom->radius = 0.5 * atof(p);
          d->counter ++;
          p=strtok(NULL," \t\n");
        }
      }
      break;
      
    case HOOMD_MASS:
      /* set mass. */
      if (d->numatoms > 0) {
        char buffer[1024];
        char *p;
        molfile_atom_t *atom;
          
        if (len==1 && ((*s == ' ') || (*s == '\n') || (*s == '\t'))) return;

        lmax=1023;
        if (len < lmax) lmax=len;
        memcpy(buffer,s,lmax);
        buffer[lmax]='\0';

        p=strtok(buffer," \t\n");
        while (p) {
          atom=d->atomlist + d->counter;
          if (atom == NULL) return;

          atom->mass = atof(p);
          d->counter ++;
          p=strtok(NULL," \t\n");
        }
      }
      break;
      
    case HOOMD_BODY:
      /* set resid from <body> tag. */
      if (d->numatoms > 0) {
        char buffer[1024];
        char *p;
        molfile_atom_t *atom;
          
        if (len==1 && ((*s == ' ') || (*s == '\n') || (*s == '\t'))) return;

        lmax=1023;
        if (len < lmax) lmax=len;
        memcpy(buffer,s,lmax);
        buffer[lmax]='\0';

        p=strtok(buffer," \t\n");
        while (p) {
          atom=d->atomlist + d->counter;
          if (atom == NULL) return;

          atom->resid = atoi(p);
          d->counter ++;
          p=strtok(NULL," \t\n");
        }
      }
      break;
      
    case HOOMD_CHARGE:
      /* set charge. */
      if (d->majv == 1 && d->minv > 2) {  /* charge is only valid from version 1.2 on */
        if (d->numatoms > 0) {
          char buffer[1024];
          char *p;
          molfile_atom_t *atom;
          
          if (len==1 && ((*s == ' ') || (*s == '\n') || (*s == '\t'))) return;

          lmax=1023;
          if (len < lmax) lmax=len;
          memcpy(buffer,s,lmax);
          buffer[lmax]='\0';

          p=strtok(buffer," \t\n");
          while (p) {
            atom=d->atomlist + d->counter;
            if (atom == NULL) return;

            atom->charge = atof(p);
            d->counter ++;
            p=strtok(NULL," \t\n");
          }
        }
      }
      break;
      
    case HOOMD_POS:
      /* only try to read coordinates. if there is storage for them. */
      if (d->ts.coords) {
        char buffer[1024];
        int lmax=1023;
        char *p;
        
        if (len < lmax) lmax=len;
        memcpy(buffer,s,lmax);
        buffer[lmax]='\0';

        p=strtok(buffer," \t\n");
        while (p) {
          if (d->counter < (3 * d->numatoms))
            d->ts.coords[d->counter] = atof(p);
          d->counter ++;
          p=strtok(NULL," \t\n");
        }
        
      }
      break;

#if vmdplugin_ABIVERSION > 10
    case HOOMD_VEL:
      if ((d->numatoms > 0) && (d->ts.velocities != NULL)) {
        char buffer[1024];
        int lmax=1023;
        char *p;
        
        if (len < lmax) lmax=len;
        memcpy(buffer,s,lmax);
        buffer[lmax]='\0';

        p=strtok(buffer," \t\n");
        while (p) {
          if (d->counter < (3 * d->numatoms))
            d->ts.velocities[d->counter] = atof(p);
          d->counter ++;
          p=strtok(NULL," \t\n");
        }
      }
      break;
#endif
      
    case HOOMD_IMAGE:
      if ((d->numatoms > 0) && (d->imagecounts != NULL)) {
        char buffer[1024];
        int lmax=1023;
        char *p;
        
        if (len < lmax) lmax=len;
        memcpy(buffer,s,lmax);
        buffer[lmax]='\0';

        p=strtok(buffer," \t\n");
        while (p) {
          d->imagecounts[d->counter] = atoi(p);
          d->counter ++;
          p=strtok(NULL," \t\n");
        }
      }
      break;
      
    default:
      break;
  }
}

/*! comment handler */
static void xml_comment(void *data, const XML_Char *s)
{
  hoomd_data_t *d=data;
  if (d==NULL) return;
#if HOOMD_DEBUG
  printf("COMMENT: %s. parsedepth=%d\n", (char *)s, d->parsedepth);
#endif
}

/*! run the XML parser to read the next line */
static int hoomd_parse_line(hoomd_data_t *data)
{   
  int done, len;
      
  data->buffer = XML_GetBuffer(data->p, MOLFILE_BIGBUFSIZ);
  done = (NULL == fgets(data->buffer,MOLFILE_BIGBUFSIZ,data->fp));

  if (!done) 
    len=strlen(data->buffer);
  else
    len=0;
      
  if (ferror(data->fp)) {
    vmdcon_printf(VMDCON_ERROR, "hoomdplugin) problem reading HOOMD"
                  " data file '%s'\n", data->filename);
    return MOLFILE_ERROR;
  }

  if (! XML_ParseBuffer(data->p, len, done)) {
    vmdcon_printf(VMDCON_ERROR, 
                  "hoomdplugin) XML syntax error at line %d:\n%s\n",
                  XML_GetCurrentLineNumber(data->p),
                  XML_ErrorString(XML_GetErrorCode(data->p)));
    return MOLFILE_ERROR;
  }
      
  if (data->parse_error > 0) {
    vmdcon_printf(VMDCON_ERROR, 
                  "hoomdplugin) XML data parse error at line %d.\n",
                  XML_GetCurrentLineNumber(data->p));
    return MOLFILE_ERROR;
  }
  
  return MOLFILE_SUCCESS;
}


/*! open the file and validate that it is indeed a HOOMD file.
 *  we also need to count the number of atoms, so we instead
 *  parse the whole first configuration including bonding info
 *  and so on and store it for later use. */
  static void *open_hoomd_read(const char *filename, const char *filetype, 
                               int *natoms) {
    FILE *fp;
    XML_Parser p;
    hoomd_data_t *data;

    fp = fopen(filename, "rb");
    if (!fp) return NULL;
  
    data = (hoomd_data_t *)calloc(1,sizeof(hoomd_data_t));
    if (data) {
      
      data->counter       =  0;
      data->numatoms      = -1;
      data->numtypes      = -1;
      data->numbonds      = -1;
      data->numangles     = -1;
      data->numdihedrals  = -1;
      data->numimpropers  = -1;
      data->numframe      = -1;
      data->numbondtypes  = -1;
      data->numangletypes = -1;
      data->numdihedraltypes = -1;
      data->numimpropertypes = -1;
      data->bondtype      = NULL;
      data->bondtypename  = NULL;
      data->angle         = NULL;
      data->angletype     = NULL;
      data->angletypename = NULL;
      data->dihedral         = NULL;
      data->dihedraltype     = NULL;
      data->dihedraltypename = NULL;
      data->dihedral         = NULL;
      data->dihedraltype     = NULL;
      data->dihedraltypename = NULL;
      data->parse_error   =  0;
      
      data->majv     = -1;
      data->minv     = -1;

      data->optflags = MOLFILE_NOOPTIONS;
      /* scaling factor for guessed diameters */
      data->mysigma  = 1.0f;
    
      p = XML_ParserCreate(NULL);
      if (!p) {
        vmdcon_printf(VMDCON_ERROR, "hoomdplugin) Could not create XML"
                      " parser for HOOMD-blue data file '%s'\n", filename);
        SAFE_FREE(data);
        fclose(fp);
        return NULL;
      }
    
      XML_SetElementHandler(p, xml_new_tag, xml_end_tag);
      XML_SetCommentHandler(p, xml_comment);
      XML_SetCharacterDataHandler(p, xml_data_block);
      XML_SetUserData(p,data);
      data->p  = p;
      data->fp = fp;
      data->filename = strdup(filename);

      /* loop through file until we have parsed the first configuration */
      do {
        if (MOLFILE_ERROR == hoomd_parse_line(data)) {
          vmdcon_printf(VMDCON_ERROR, "hoomdplugin) XML Parse error "
                      "while reading HOOMD-blue data file '%s'\n", filename);
          XML_ParserFree(data->p);
          data->p = NULL;
          SAFE_FREE(data->filename);
          fclose(data->fp);
          SAFE_FREE(data);
          return NULL;
        }
      } while (!feof(fp) && !data->doneframe);

      if ( data->majv > HOOMD_FORMAT_MAJV ) {
        vmdcon_printf(VMDCON_ERROR, "hoomdplugin) Encountered incompatible "
                      "HOOMD-blue data file format version '%d.%d.\n", 
                      data->majv, data->minv);
        vmdcon_printf(VMDCON_ERROR, "hoomdplugin) This plugin supports only "
                      "HOOMD-blue data files up to version '%d.%d'.\n",
                      HOOMD_FORMAT_MAJV, HOOMD_FORMAT_MINV);
        XML_ParserFree(data->p);
        data->p = NULL;
        SAFE_FREE(data->filename);
        fclose(data->fp);
        SAFE_FREE(data);
        return NULL;
      } else {
        if ( (data->majv == HOOMD_FORMAT_MAJV) && 
             (data->minv > HOOMD_FORMAT_MINV) ) {
          vmdcon_printf(VMDCON_WARN, "hoomdplugin) Encountered newer HOOMD-blue "
                        "data file format version '%d.%d'.\n",
                        data->majv, data->minv);
          vmdcon_printf(VMDCON_WARN, "hoomdplugin) This plugin supports HOOMD-blue "
                        "data files up to version '%d.%d'. Continuing...\n", 
                        HOOMD_FORMAT_MAJV, HOOMD_FORMAT_MINV);
        }
      }

      if (data->numatoms < 0) {
        vmdcon_printf(VMDCON_ERROR, "hoomdplugin) Could not determine "
                      "number of atoms in HOOMD-blue data file '%s'\n", filename);
        XML_ParserFree(data->p);
        data->p = NULL;
        SAFE_FREE(data->filename);
        fclose(data->fp);
        SAFE_FREE(data);
        return NULL;
      }

      /* reset parsing */
      XML_ParserFree(p);
      data->p = NULL;
      rewind(fp);
    
      data->counter  = 0;
      data->numframe = 0;
      *natoms=data->numatoms;
    }
  
    return data;
  }

static int read_hoomd_structure(void *mydata, int *optflags, 
                                molfile_atom_t *atoms) {
  molfile_atom_t *a;
  XML_Parser p;
  const char *envvar;
  int i;
  hoomd_data_t *data = (hoomd_data_t *)mydata;
  
  data->parsedepth = 0;
  data->counter    = 0;
  data->numframe   = 0;
  data->doneframe  = 0;
  data->atomlist   = atoms;
  SAFE_CALLOC(data->imagecounts,   int,   3*data->numatoms);
  SAFE_CALLOC(data->bondtypename,  char*, 1);  /* is grown on demand */
  SAFE_CALLOC(data->angletypename, char*, 1);  /* is grown on demand */
  SAFE_CALLOC(data->dihedraltypename, char*, 1);  /* is grown on demand */
  SAFE_CALLOC(data->impropertypename, char*, 1);  /* is grown on demand */
  if (data->numbonds > 0) {
    SAFE_CALLOC(data->from,      int, data->numbonds);
    SAFE_CALLOC(data->to,        int, data->numbonds);
    SAFE_CALLOC(data->bondtype,  int, data->numbonds);
  }
  if (data->numangles > 0) {
    SAFE_CALLOC(data->angle,     int, data->numangles * 3);
    SAFE_CALLOC(data->angletype, int, data->numangles);
  }
  if (data->numdihedrals > 0) {
    SAFE_CALLOC(data->dihedral,     int, data->numdihedrals * 4);
    SAFE_CALLOC(data->dihedraltype, int, data->numdihedrals);
  }
  if (data->numimpropers > 0) {
    SAFE_CALLOC(data->improper,     int, data->numimpropers * 4);
    SAFE_CALLOC(data->impropertype, int, data->numimpropers);
  }
  SAFE_CALLOC(data->ts.coords,float,3*data->numatoms);
  data->ts.A = data->ts.B = data->ts.C = 0.0f;
  data->ts.alpha = data->ts.beta = data->ts.gamma = 90.0f;
  
#if vmdplugin_ABIVERSION > 10
  data->ts_meta.count = -1;
  data->ts_meta.has_velocities = 0;
  SAFE_CALLOC(data->ts.velocities,float,3*data->numatoms);
#endif
  p = XML_ParserCreate(NULL);
  if (!p) {
    vmdcon_printf(VMDCON_ERROR, "hoomdplugin) Could not create XML"
                  " parser for HOOMD-blue data file '%s'\n", data->filename);
    return MOLFILE_ERROR;
  }

  XML_SetElementHandler(p, xml_new_tag, xml_end_tag);
  XML_SetCommentHandler(p, xml_comment);
  XML_SetCharacterDataHandler(p,xml_data_block);
  XML_SetUserData(p,data);
  data->p = p;

  /* initialize atomdata with typical defaults */
  for (i=0, a=atoms; i < data->numatoms; ++i, ++a) {
    a->radius =  1.0f;
    a->mass =    1.0f;
    a->resname[0]='\0';
    a->resid=0;
    a->segid[0]='\0';
    a->chain[0]='\0';
  }
    
  /* read the first configuration again, but keep the data this time. */
  do {
    if (MOLFILE_ERROR == hoomd_parse_line(data)) {
      XML_ParserFree(data->p);
      data->p = NULL;
      return MOLFILE_ERROR;
    } 
  } while (!feof(data->fp) && !data->doneframe);

  /* allow overriding hardcoded sigma values from environment */
  envvar = getenv("VMDHOOMDSIGMA");
  if (envvar) data->mysigma = atof(envvar);

  /* fix up settings */
  for (i=0, a=atoms; i < data->numatoms; ++i, ++a) {

    if (!(data->optflags & MOLFILE_RADIUS)) {
      /* use vizsigma parameter to adjust guessed particle radii,
         but only for xml files that don't have a diameter section. */
      a->radius = data->mysigma * get_pte_vdw_radius(a->atomicnumber);
    }

    if (!(data->optflags & MOLFILE_MASS)) {
      /* guess atom mass, if not provided */
      a->mass = get_pte_mass(a->atomicnumber);
    }
  }
  *optflags = data->optflags | MOLFILE_RADIUS | MOLFILE_MASS;
    
  data->numframe=1;

  return MOLFILE_SUCCESS;
}

#if vmdplugin_ABIVERSION > 10
/***********************************************************/
static int read_timestep_metadata(void *mydata,
                                  molfile_timestep_metadata_t *meta) {
  hoomd_data_t *data = (hoomd_data_t *)mydata;
  
  meta->count = -1;
  meta->has_velocities = data->ts_meta.has_velocities;
  if (meta->has_velocities) {
    vmdcon_printf(VMDCON_INFO, "hoomdplugin) Importing velocities.\n");
  }
  return MOLFILE_SUCCESS;
}
#endif



#if vmdplugin_ABIVERSION > 14
static int read_hoomd_bonds(void *v, int *nbonds, int **fromptr, int **toptr, 
                            float **bondorder, int **bondtype, 
                            int *nbondtypes, char ***bondtypename) {
  hoomd_data_t *data = (hoomd_data_t *)v;

  *nbonds = data->numbonds;
  *bondorder = NULL;

  if (data->numbonds > 0) {
    *fromptr      = data->from;
    *toptr        = data->to;
    *bondtype     = data->bondtype;
    *nbondtypes   = data->numbondtypes;
    *bondtypename = data->bondtypename;
  } else {
    *fromptr      = NULL;
    *toptr        = NULL;
    *bondtype     = NULL;
    *nbondtypes   = 0;
    *bondtypename = NULL;
    vmdcon_printf(VMDCON_WARN,
                  "hoomdplugin) no bonds defined in data file.\n");
  }
  return MOLFILE_SUCCESS;
}
#else
static int read_hoomd_bonds(void *v, int *nbonds, int **fromptr, int **toptr,
                            float **bondorder) {
  hoomd_data_t *data = (hoomd_data_t *)v;

  *nbonds = data->numbonds;
  *bondorder = NULL;

  if (data->numbonds > 0) {
    *fromptr      = data->from;
    *toptr        = data->to;
  } else {
    *fromptr      = NULL;
    *toptr        = NULL;
    vmdcon_printf(VMDCON_WARN,
                  "hoomdplugin) no bonds defined in data file.\n");
  }
  return MOLFILE_SUCCESS;
}
#endif

#if vmdplugin_ABIVERSION > 15
static int read_hoomd_angles(void *v, int *numangles, int **angles, 
                             int **angletypes, int *numangletypes, 
                             char ***angletypenames, int *numdihedrals,
                             int **dihedrals, int **dihedraltypes, 
                             int *numdihedraltypes, char ***dihedraltypenames,
                             int *numimpropers, int **impropers, 
                             int **impropertypes, int *numimpropertypes, 
                             char ***impropertypenames, int *numcterms, 
                             int **cterms, int *ctermcols, int *ctermrows) {
                               
  hoomd_data_t *data = (hoomd_data_t *)v;

  *numangles         = 0;
  *angles            = NULL;
  *angletypes        = NULL;
  *numangletypes     = 0;
  *angletypenames    = NULL;
  *numdihedrals      = 0;
  *dihedrals         = NULL;
  *dihedraltypes     = NULL;
  *numdihedraltypes  = 0;
  *dihedraltypenames = NULL;
  *numimpropers      = 0;
  *impropers         = NULL;
  *impropertypes     = NULL;
  *numimpropertypes  = 0;
  *impropertypenames = NULL;
  *numcterms         = 0;           /* hoomd does not support CMAP */
  *cterms            = NULL;
  *ctermrows         = 0;
  *ctermcols         = 0;
  
  if (data->numangles > 0) {
    *numangles      = data->numangles;
    *angles         = data->angle;
    *angletypes     = data->angletype;
    *numangletypes  = data->numangletypes;
    *angletypenames = data->angletypename;
  } else {
    vmdcon_printf(VMDCON_INFO,
                  "hoomdplugin) no angles defined in data file.\n");
  }
  if (data->numdihedrals > 0) {
    *numdihedrals      = data->numdihedrals;
    *dihedrals         = data->dihedral;
    *dihedraltypes     = data->dihedraltype;
    *numdihedraltypes  = data->numdihedraltypes;
    *dihedraltypenames = data->dihedraltypename;
  } else {
    vmdcon_printf(VMDCON_INFO,
                  "hoomdplugin) no dihedrals defined in data file.\n");
  }
  if (data->numimpropers > 0) {
    *numimpropers      = data->numimpropers;
    *impropers         = data->improper;
    *impropertypes     = data->impropertype;
    *numimpropertypes  = data->numimpropertypes;
    *impropertypenames = data->impropertypename;
  } else {
    vmdcon_printf(VMDCON_INFO,
                  "hoomdplugin) no impropers defined in data file.\n");
  }
  return MOLFILE_SUCCESS;
}
#else
static int read_hoomd_angles(void *v, int *numangles, int **angles, 
                             double **angleforces, int *numdihedrals, 
                             int **dihedrals, double **dihedralforces,
                             int *numimpropers, int **impropers, 
                             double **improperforces, int *numcterms,
                             int **cterms, int *ctermcols, int *ctermrows,  
                             double **ctermforces) {
  hoomd_data_t *data = (hoomd_data_t *)v;

  *numangles      = 0;
  *angles         = NULL;
  *angleforces    = NULL;
  *numdihedrals   = 0;          /* we currently only support angles. */
  *dihedrals      = NULL;
  *dihedralforces = NULL;
  *numimpropers   = 0;
  *impropers      = NULL;
  *improperforces = NULL;
  *numcterms      = 0;
  *cterms         = NULL;
  *ctermrows      = 0;
  *ctermcols      = 0;
  *ctermforces    = NULL;
  
  if (data->numangles > 0) {
    *numangles      = data->numangles;
    *angles         = data->angle;
  } else {
    vmdcon_printf(VMDCON_INFO,
                  "hoomdplugin) no angles defined in data file.\n");
  }
  if (data->numdihedrals > 0) {
    *numdihedrals      = data->numdihedrals;
    *dihedrals         = data->dihedral;
  } else {
    vmdcon_printf(VMDCON_INFO,
                  "hoomdplugin) no dihedrals defined in data file.\n");
  }
  if (data->numimpropers > 0) {
    *numimpropers      = data->numimpropers;
    *impropers         = data->improper;
  } else {
    vmdcon_printf(VMDCON_INFO,
                  "hoomdplugin) no impropers defined in data file.\n");
  }
  return MOLFILE_SUCCESS;
}
#endif

static int read_hoomd_timestep(void *mydata, int natoms, 
                               molfile_timestep_t *ts) {
  int i;
  hoomd_data_t *data;
  
  data=(hoomd_data_t *)mydata;

  if (data->parse_error) return MOLFILE_ERROR;
  if (data->p == NULL) return MOLFILE_ERROR;
  
  if (data->numframe > 1) {
    /* read the next configuration. */
    data->doneframe =  0;
    do {
      if (MOLFILE_ERROR == hoomd_parse_line(data)) {
        XML_ParserFree(data->p);
        data->p = NULL;
        return MOLFILE_ERROR;
      } 
      if (data->parse_error) return MOLFILE_ERROR;
    } while (!feof(data->fp) && !data->doneframe);
    if (feof(data->fp)) return MOLFILE_ERROR;
  }
  data->numframe++;
  
  if (ts != NULL) { 
    /* only save coords if we're given a timestep pointer, */
    /* otherwise assume that VMD wants us to skip past it. */
    for (i=0; i<natoms; ++i) {
      ts->coords[3*i+0] = data->ts.coords[3*i+0]
        + data->ts.A * data->imagecounts[3*i+0];
      ts->coords[3*i+1] = data->ts.coords[3*i+1]
        + data->ts.B * data->imagecounts[3*i+1];
      ts->coords[3*i+2] = data->ts.coords[3*i+2]
        + data->ts.C * data->imagecounts[3*i+2];
    }
#if vmdplugin_ABIVERSION > 10    
    /* copy velocities */
    if (ts->velocities != NULL) {
      for (i=0; i<natoms; ++i) {
        ts->velocities[3*i+0] = data->ts.velocities[3*i+0];
        ts->velocities[3*i+1] = data->ts.velocities[3*i+1];
        ts->velocities[3*i+2] = data->ts.velocities[3*i+2];
      }
    }
#endif
    ts->A = data->ts.A;
    ts->B = data->ts.B;
    ts->C = data->ts.C;
    ts->alpha = data->ts.alpha;
    ts->beta  = data->ts.beta;
    ts->gamma = data->ts.gamma;
  }
  return MOLFILE_SUCCESS;
}
    
static void close_hoomd_read(void *mydata) {
  hoomd_data_t *data = (hoomd_data_t *)mydata;
  int i;
  
  if (data->p != NULL) XML_ParserFree(data->p);

  fclose(data->fp);

  SAFE_FREE(data->imagecounts);
  SAFE_FREE(data->from);
  SAFE_FREE(data->to);
  SAFE_FREE(data->bondtype);
  for (i=0; i < data->numbondtypes; ++i) {
    SAFE_FREE(data->bondtypename[i]);
  }
  SAFE_FREE(data->bondtypename);
  SAFE_FREE(data->angle);
  SAFE_FREE(data->angletype);
  for (i=0; i < data->numangletypes; ++i) {
    SAFE_FREE(data->angletypename[i]);
  }
  SAFE_FREE(data->angletypename);
  SAFE_FREE(data->dihedral);
  SAFE_FREE(data->dihedraltype);
  for (i=0; i < data->numdihedraltypes; ++i) {
    SAFE_FREE(data->dihedraltypename[i]);
  }
  SAFE_FREE(data->dihedraltypename);
  SAFE_FREE(data->improper);
  SAFE_FREE(data->impropertype);
  for (i=0; i < data->numimpropertypes; ++i) {
    SAFE_FREE(data->impropertypename[i]);
  }
  SAFE_FREE(data->impropertypename);
  SAFE_FREE(data->ts.coords);
  SAFE_FREE(data->ts.velocities);
  SAFE_FREE(data->filename);
  SAFE_FREE(data);
}

/*! open hoomd data file for writing.
 * this will also write the XML identfier tag and the opening 
 * <hoomd> and <configuration> elements */
static void *open_hoomd_write(const char *filename, const char *filetype, 
                              int natoms) {
  FILE *fd;
  hoomd_data_t *data;
  time_t mytime;

  mytime = time(NULL);
  

  fd = fopen(filename, "w");
  if (!fd) { 
    vmdcon_printf(VMDCON_ERROR, "hoomdplugin) Unable to open HOOMD-blue data file %s "
                  "for writing\n", filename);
    return NULL;
  }
  
  data = (hoomd_data_t *)malloc(sizeof(hoomd_data_t));
  data->numatoms = natoms;
  data->filename = strdup(filename);
  data->fp   = fd;
  data->atomlist      = NULL;
  data->numbonds      = 0;
  data->to            = NULL;
  data->from          = NULL;
  data->bondtype      = NULL;
  data->numbondtypes  = 0;
  data->bondtypename  = NULL;
  data->numangles     = 0;
  data->angle         = NULL;
  data->angletype     = NULL;
  data->numangletypes = 0;
  data->angletypename = NULL;
  data->numdihedrals     = 0;
  data->dihedral         = NULL;
  data->dihedraltype     = NULL;
  data->numdihedraltypes = 0;
  data->dihedraltypename = NULL;
  data->numimpropers     = 0;
  data->improper         = NULL;
  data->impropertype     = NULL;
  data->numimpropertypes = 0;
  data->impropertypename = NULL;
  data->numframe      = 0;
  data->imagecounts   = (int *)malloc(sizeof(int)*3*(data->numatoms));

  fputs("<?xml version =\"1.0\" encoding =\"UTF-8\" ?>\n",fd);
  fprintf(fd,"<%s version=\"%d.%d\">\n", hoomd_tag_name[HOOMD_XML],
          HOOMD_FORMAT_MAJV, HOOMD_FORMAT_MINV);
  fprintf(fd, "<!-- generated by VMD on: %s", ctime(&mytime));
  fprintf(fd, " %s plugin v%d.%d by %s -->\n", plugin.prettyname,
          plugin.majorv, plugin.minorv, plugin.author);

  fprintf(data->fp,"<%s time_step=\"%d\" natoms=\"%d\">\n",
          hoomd_tag_name[HOOMD_CONFIG], data->numframe, data->numatoms);

  return data;
}


/* some macros for consistency and convenience */
#define SECTION_OPEN(tag,num) \
  fprintf(data->fp, "<%s num=\"%d\">\n", hoomd_tag_name[tag], num)

#define SECTION_CLOSE(tag) \
  fprintf(data->fp, "</%s>\n", hoomd_tag_name[tag])
  
#define SECTION_WRITE_ATOMIC(tag,fmt,val) \
  SECTION_OPEN(tag,numatoms);             \
  for (i=0; i < numatoms; ++i)            \
    fprintf(data->fp,fmt, val);           \
  SECTION_CLOSE(tag)

/*! write topology data to hoomd data file.
 */
static int write_hoomd_structure(void *mydata, int optflags, 
                                 const molfile_atom_t *atoms) {
  int i, numatoms;
  hoomd_data_t *data = (hoomd_data_t *)mydata;
  numatoms = data->numatoms;


  /* write fields we know about */
  /* required by molfile: */
  SECTION_WRITE_ATOMIC(HOOMD_TYPE,"%s\n",atoms[i].type);
  SECTION_WRITE_ATOMIC(HOOMD_BODY,"%d\n",atoms[i].resid);

  /* optional: */
  if (optflags & MOLFILE_RADIUS) {
    SECTION_WRITE_ATOMIC(HOOMD_DIAMETER,"%f\n",2.0f*atoms[i].radius);
  }
  
  if (optflags & MOLFILE_MASS) {
    SECTION_WRITE_ATOMIC(HOOMD_MASS,"%f\n",atoms[i].mass);
  }
  
  if (optflags & MOLFILE_CHARGE) {
    SECTION_WRITE_ATOMIC(HOOMD_CHARGE,"%f\n",atoms[i].charge);
  }
  
  if ((data->numbonds > 0) && (data->from != NULL) && (data->to != NULL)) {
    SECTION_OPEN(HOOMD_BOND, data->numbonds);
    if (data->bondtype != NULL) {
      if (data->bondtypename != NULL) {
        /* case 1: symbolic bond types available */
        for (i=0; i < data->numbonds; ++i) {
          if (data->bondtype[i] < 0) {
            /* case 1a: symbolic bond types not assigned */
            fprintf(data->fp,"unkown %d %d\n", data->from[i]-1, data->to[i]-1);
          } else {
            fprintf(data->fp,"%s %d %d\n", data->bondtypename[data->bondtype[i]],
                    data->from[i]-1, data->to[i]-1);
          }
        }
      } else {
        /* case 2: only numerical bond types available */
        for (i=0; i < data->numbonds; ++i) {
          fprintf(data->fp,"bondtype%d %d %d\n", data->bondtype[i], 
                  data->from[i]-1, data->to[i]-1);
        }
      }
    } else {
      /* case 3: no bond type info available. */
      for (i=0; i < data->numbonds; ++i) {
        fprintf(data->fp,"bond %d %d\n", data->from[i]-1, data->to[i]-1);
      }
    }
    SECTION_CLOSE(HOOMD_BOND);
  }
  
  if ( (data->numangles > 0) && (data->angle != NULL) ) {
    SECTION_OPEN(HOOMD_ANGLE,data->numangles);
    if (data->angletype != NULL) {
      if (data->angletypename != NULL) {
        /* case 1: symbolic angle types available */
        for (i=0; i < data->numangles; ++i) {
          if (data->angletype[i] < 0) {
            /* case 1a: symbolic angle types not assigned */
            fprintf(data->fp,"unkown %d %d %d\n", data->angle[3*i]-1, 
                    data->angle[3*i+1]-1, data->angle[3*i+2]-1);
          } else {
            fprintf(data->fp,"%s %d %d %d\n", 
                    data->angletypename[data->angletype[i]], 
                    data->angle[3*i]-1, data->angle[3*i+1]-1, 
                    data->angle[3*i+2]-1);
          }
        }
      } else {
        /* case 2: only numerical angle types available */
        for (i=0; i < data->numangles; ++i) {
          fprintf(data->fp,"angletype%d %d %d %d\n", 
                  data->angletype[i], data->angle[3*i]-1, 
                  data->angle[3*i+1]-1, data->angle[3*i+2]-1);
        }
      }
    } else {
      /* case 3: no angle type info available. */
      for (i=0; i < data->numangles; ++i) {
        fprintf(data->fp,"angle %d %d %d\n", data->angle[3*i]-1, 
                data->angle[3*i+1]-1, data->angle[3*i+2]-1);
      }
    }
    SECTION_CLOSE(HOOMD_ANGLE);
  }

  if ( (data->numdihedrals > 0) && (data->dihedral != NULL) ) {
    SECTION_OPEN(HOOMD_DIHEDRAL, data->numdihedrals);
    if (data->dihedraltype != NULL) {
      if (data->dihedraltypename != NULL) {
        /* case 1: symbolic dihedral types available */
        for (i=0; i < data->numdihedrals; ++i) {
          if (data->dihedraltype[i] < 0) {
            /* case 1a: symbolic dihedral types not assigned */
            fprintf(data->fp,"unkown %d %d %d %d\n", data->dihedral[4*i]-1, 
                    data->dihedral[4*i+1]-1, data->dihedral[4*i+2]-1,
                    data->dihedral[4*i+3]-1);
          } else {
            fprintf(data->fp,"%s %d %d %d %d\n", 
                    data->dihedraltypename[data->dihedraltype[i]], 
                    data->dihedral[4*i]-1,   data->dihedral[4*i+1]-1, 
                    data->dihedral[4*i+2]-1, data->dihedral[4*i+3]-1);
          }
        }
      } else {
        /* case 2: only numerical dihedral types available */
        for (i=0; i < data->numdihedrals; ++i) {
          fprintf(data->fp,"dihedraltype%d %d %d %d %d\n", 
                  data->dihedraltype[i],   data->dihedral[4*i]-1, 
                  data->dihedral[4*i+1]-1, data->dihedral[4*i+2]-1,
                  data->dihedral[4*i+3]-1);
        }
      }
    } else {
      /* case 3: no dihedral type info available. */
      for (i=0; i < data->numdihedrals; ++i) {
        fprintf(data->fp,"dihedral %d %d %d %d\n", 
                data->dihedral[4*i]-1,   data->dihedral[4*i+1]-1, 
                data->dihedral[4*i+2]-1, data->dihedral[4*i+3]-1);
      }
    }
    SECTION_CLOSE(HOOMD_DIHEDRAL);
  }

  if ( (data->numimpropers > 0) && (data->improper != NULL) ) {
    SECTION_OPEN(HOOMD_IMPROPER, data->numimpropers);
    if (data->impropertype != NULL) {
      if (data->impropertypename != NULL) {
        /* case 1: symbolic improper types available */
        for (i=0; i < data->numimpropers; ++i) {
          if (data->impropertype[i] < 0) {
            /* case 1a: symbolic improper types not assigned */
            fprintf(data->fp,"unkown %d %d %d %d\n", data->improper[4*i]-1, 
                    data->improper[4*i+1]-1, data->improper[4*i+2]-1,
                    data->improper[4*i+3]-1);
          } else {
            fprintf(data->fp,"%s %d %d %d %d\n", 
                    data->impropertypename[data->impropertype[i]], 
                    data->improper[4*i]-1,   data->improper[4*i+1]-1, 
                    data->improper[4*i+2]-1, data->improper[4*i+3]-1);
          }
        }
      } else {
        /* case 2: only numerical improper types available */
        for (i=0; i < data->numimpropers; ++i) {
          fprintf(data->fp,"impropertype%d %d %d %d %d\n", 
                  data->impropertype[i],   data->improper[4*i]-1, 
                  data->improper[4*i+1]-1, data->improper[4*i+2]-1,
                  data->improper[4*i+3]-1);
        }
      }
    } else {
      /* case 3: no improper type info available. */
      for (i=0; i < data->numimpropers; ++i) {
        fprintf(data->fp,"improper %d %d %d %d\n", 
                data->improper[4*i]-1,   data->improper[4*i+1]-1, 
                data->improper[4*i+2]-1, data->improper[4*i+3]-1);
      }
    }
    SECTION_CLOSE(HOOMD_IMPROPER);
  }
  return MOLFILE_SUCCESS;
}
/* clean up, part 1 */
#undef SECTION_WRITE_ATOMIC


#if vmdplugin_ABIVERSION > 14
static int write_hoomd_bonds(void *v, int nbonds, int *fromptr, int *toptr, 
                             float *bondorder, int *bondtype, 
                             int nbondtypes, char **bondtypename) {
  hoomd_data_t *data = (hoomd_data_t *)v;
  int i;

  data->numbonds=0;
  data->numbondtypes=0;
  data->bondtype=NULL;

  /* save info until we actually write out the structure data */
  if ( (nbonds > 0) && (fromptr != NULL) && (toptr != NULL) ) {
    data->numbonds = nbonds;
    SAFE_CALLOC(data->from, int, nbonds);
    memcpy(data->from, fromptr, nbonds * sizeof(int));
    SAFE_CALLOC(data->to,   int, nbonds);
    memcpy(data->to,   toptr,   nbonds * sizeof(int));
    if (bondtype != NULL) {
      SAFE_CALLOC(data->bondtype, int, nbonds);
      memcpy(data->bondtype, bondtype, nbonds * sizeof(int));
    }
  }
  if (nbondtypes > 0) {
    data->numbondtypes=nbondtypes;
    if (bondtypename != NULL) {
      SAFE_CALLOC(data->bondtypename, char *, nbondtypes+1);
      for (i=0; i < nbondtypes; ++i) {
        data->bondtypename[i] = strdup(bondtypename[i]);
      }
    }
  }

  return MOLFILE_SUCCESS;
}
#else
static int write_hoomd_bonds(void *v, int nbonds, int *fromptr, int *toptr, float *bondorder) {
  hoomd_data_t *data = (hoomd_data_t *)v;

  data->numbonds=0;
  data->numbondtypes=0;
  data->bondtype=NULL;

  /* save info until we actually write out the structure data */
  if ( (nbonds > 0) && (fromptr != NULL) && (toptr != NULL) ) {
    data->numbonds = nbonds;
    SAFE_CALLOC(data->from, int, nbonds);
    memcpy(data->from, fromptr, nbonds * sizeof(int));
    SAFE_CALLOC(data->to,   int, nbonds);
    memcpy(data->to,   toptr,   nbonds * sizeof(int));
  }

  return MOLFILE_SUCCESS;
}
#endif

#if vmdplugin_ABIVERSION > 15
static int write_hoomd_angles(void * v, int numangles, const int *angles,
                              const int *angletypes, int numangletypes,
                              const char **angletypenames, int numdihedrals, 
                              const int *dihedrals, const int *dihedraltypes,
                              int numdihedraltypes, const char **dihedraltypenames,
                              int numimpropers, const int *impropers, 
                              const int *impropertypes, int numimpropertypes, 
                              const char **impropertypenames, int numcterms, 
                              const int *cterm, int ctermcols, int ctermrows) {
  hoomd_data_t *data = (hoomd_data_t *)v;
  int i;
  
  /* save info until we actually write out the structure file */
  data->numangles    = numangles;
  data->numdihedrals = numdihedrals;
  data->numimpropers = numimpropers;

  if (data->numangles > 0) {
    SAFE_CALLOC(data->angle, int, 3*data->numangles);
    memcpy(data->angle, angles, 3*(data->numangles)*sizeof(int));
  }
  if (data->numdihedrals > 0) {
    SAFE_CALLOC(data->dihedral, int, 4*data->numdihedrals);
    memcpy(data->dihedral, dihedrals, 4*(data->numdihedrals)*sizeof(int));
  }
  if (data->numimpropers > 0) {
    SAFE_CALLOC(data->improper, int, 4*data->numimpropers);
    memcpy(data->improper, impropers, 4*(data->numimpropers)*sizeof(int));
  }

  if (angletypes != NULL) {
    SAFE_CALLOC(data->angletype, int, data->numangles);
    memcpy(data->angletype, angletypes, (data->numangles)*sizeof(int));
  }
  if (dihedraltypes != NULL) {
    SAFE_CALLOC(data->dihedraltype, int, data->numdihedrals);
    memcpy(data->dihedraltype, dihedraltypes, (data->numdihedrals)*sizeof(int));
  }
  if (impropertypes != NULL) {
    SAFE_CALLOC(data->impropertype, int, data->numimpropers);
    memcpy(data->impropertype, impropertypes, (data->numimpropers)*sizeof(int));
  }

  data->numangletypes = numangletypes;
  if (data->numangletypes > 0) {
    if (angletypenames != NULL) {
      SAFE_CALLOC(data->angletypename, char *, (data->numangletypes)+1);
      for (i=0; i < data->numangletypes; ++i) {
        data->angletypename[i] = strdup(angletypenames[i]);
      }
    }
  }
  data->numdihedraltypes = numdihedraltypes;
  if (data->numdihedraltypes > 0) {
    if (dihedraltypenames != NULL) {
      SAFE_CALLOC(data->dihedraltypename, char *, (data->numdihedraltypes)+1);
      for (i=0; i < data->numdihedraltypes; ++i) {
        data->dihedraltypename[i] = strdup(dihedraltypenames[i]);
      }
    }
  }

  data->numimpropertypes = numimpropertypes;
  if (data->numimpropertypes > 0) {
    if (impropertypenames != NULL) {
      SAFE_CALLOC(data->impropertypename, char *, (data->numimpropertypes)+1);
      for (i=0; i < data->numimpropertypes; ++i) {
        data->impropertypename[i] = strdup(impropertypenames[i]);
      }
    }
  }

#if 0                           /* the rest is not yet supported. */
  data->numcterms    = numcterms;
  if (data->numcterms > 0) {

    data->cterms = (int *) malloc(8*data->numcterms*sizeof(int));
    memcpy(data->cterms, cterms, 8*data->numcterms*sizeof(int));
  }
#endif
  return MOLFILE_SUCCESS;
}

#else
static int write_hoomd_angles(void * v, int numangles, const int *angles,
                        const double *angleforces, int numdihedrals, 
                        const int *dihedrals, const double *dihedralforces,
                        int numimpropers, const int *impropers, 
                        const double *improperforces, int numcterms,   
                        const int *cterms, int ctermcols, int ctermrows, 
                        const double *ctermforces) {
    hoomd_data_t *data = (hoomd_data_t *)v;
  /* save info until we actually write out the structure file */
  data->numangles    = numangles;
  if (data->numangles > 0) {
    data->angle = (int *) malloc(3*data->numangles*sizeof(int));
    memcpy(data->angle, angles, 3*data->numangles*sizeof(int));
  }

  data->numdihedrals = numdihedrals;
  data->numimpropers = numimpropers;
  data->numcterms    = numcterms;

  if (data->numdihedrals > 0) {
    data->dihedrals = (int *) malloc(4*data->numdihedrals*sizeof(int));
    memcpy(data->dihedrals, dihedrals, 4*data->numdihedrals*sizeof(int));
  }
  
    data->impropers = (int *) malloc(4*data->numimpropers*sizeof(int));
    memcpy(data->impropers, impropers, 4*data->numimpropers*sizeof(int));
  }
#if 0                           /* XXX not yet supported. */
  if (data->numcterms > 0) {

    data->cterms = (int *) malloc(8*data->numcterms*sizeof(int));
    memcpy(data->cterms, cterms, 8*data->numcterms*sizeof(int));
  }
#endif
  return MOLFILE_SUCCESS;
}
#endif


static int write_hoomd_timestep(void *mydata, const molfile_timestep_t *ts) {
  hoomd_data_t *data = (hoomd_data_t *)mydata; 
  const float *pos;
  float px, py, pz;
  int i, *img, numatoms;
  numatoms = data->numatoms;

  if (data->numframe != 0) {
    fprintf(data->fp,"<%s time_step=\"%d\" natoms=\"%d\" dimensions=\"3\">\n", 
            hoomd_tag_name[HOOMD_CONFIG],
#if vmdplugin_ABIVERSION > 10
            (int)(ts->physical_time+0.5f),
#else
            data->numframe,
#endif
            numatoms);
  }

  fprintf(data->fp, "<%s lx=\"%f\" ly=\"%f\" lz=\"%f\" />\n", 
          hoomd_tag_name[HOOMD_BOX], ts->A, ts->B, ts->C);

#define HOOMD_CONV_POS(crd, box, p, i)  \
  if (fabsf(box) > SMALL) {            \
    float tmp;                          \
    tmp = floorf(crd / box + 0.5f);     \
    p =  crd - tmp*box;                 \
    i = (int) tmp;                      \
  } else {                              \
    p = crd;                            \
    i = 0.0f;                           \
  }

  SECTION_OPEN(HOOMD_POS, numatoms);
  pos = ts->coords;
  img = data->imagecounts;
  for (i = 0; i < numatoms; ++i) {
    HOOMD_CONV_POS(*pos, ts->A, px, *img);
    ++pos; ++img;
    HOOMD_CONV_POS(*pos, ts->B, py, *img);
    ++pos; ++img;
    HOOMD_CONV_POS(*pos, ts->C, pz, *img);
    ++pos; ++img;

    fprintf(data->fp, "%.6f %.6f %.6f\n", px, py, pz);
  }
  SECTION_CLOSE(HOOMD_POS);

  /* only print image section if we have (some) box info. */
  if ( (fabsf(ts->A)+fabsf(ts->B)+fabsf(ts->C)) > SMALL) {
    SECTION_OPEN(HOOMD_IMAGE,numatoms);
    img = data->imagecounts;
    for (i = 0; i < numatoms; ++i) {
      fprintf(data->fp, "%d %d %d\n", img[0], img[1], img[2]);
      img += 3;
    }
    SECTION_CLOSE(HOOMD_IMAGE);
  }

#if vmdplugin_ABIVERSION > 10
  if (ts->velocities != NULL) {
    const float *vel;

    SECTION_OPEN(HOOMD_VEL, numatoms);
    vel = ts->velocities;
    for (i = 0; i < numatoms; ++i) {
      fprintf(data->fp, "%.6f %.6f %.6f\n", vel[0], vel[1], vel[2]);
      vel += 3;
    }
    SECTION_CLOSE(HOOMD_VEL);
  }
#endif

  SECTION_CLOSE(HOOMD_CONFIG);
  data->numframe ++;

  return MOLFILE_SUCCESS;
}


static void close_hoomd_write(void *mydata) {
  hoomd_data_t *data = (hoomd_data_t *)mydata;
  int i;
  
  /* we have not yet written any positions, so the configuration tag is still open. */
  if (data->numframe == 0) {
    SECTION_CLOSE(HOOMD_CONFIG);
  }
  

  SECTION_CLOSE(HOOMD_XML);
  fclose(data->fp);

  if (data->numbonds > 0) {
    free(data->from);
    free(data->to);
    if (data->bondtype != NULL) {
      free(data->bondtype);
    }
  }
  if (data->numbondtypes > 0) {
    if (data->bondtypename != NULL) {
      for (i=0; i < data->numbondtypes; ++i) {
        free(data->bondtypename[i]);
      }
      free(data->bondtypename);
    }
  }
  
  if (data->numangles > 0) {
    free(data->angle);
    if (data->angletype != NULL) {
      free(data->angletype);
    }
  }
  if (data->numangletypes > 0) {
    if (data->angletypename != NULL) {
      for (i=0; i < data->numangletypes; ++i) {
        free(data->angletypename[i]);
      }
      free(data->angletypename);
    }
  }

  if (data->numdihedrals > 0) {
    free(data->dihedral);
    if (data->dihedraltype != NULL) {
      free(data->dihedraltype);
    }
  }
  if (data->numdihedraltypes > 0) {
    if (data->dihedraltypename != NULL) {
      for (i=0; i < data->numdihedraltypes; ++i) {
        free(data->dihedraltypename[i]);
      }
      free(data->dihedraltypename);
    }
  }

  if (data->numimpropers > 0) {
    free(data->improper);
    if (data->impropertype != NULL) {
      free(data->impropertype);
    }
  }
  if (data->numimpropertypes > 0) {
    if (data->impropertypename != NULL) {
      for (i=0; i < data->numimpropertypes; ++i) {
        free(data->impropertypename[i]);
      }
      free(data->impropertypename);
    }
  }

  
  if (data->atomlist) {
    free(data->atomlist);
  }
  free(data->imagecounts);
  free(data->filename);
  free(data);
}
/* cleanup, part 2 */
#undef SECTION_OPEN
#undef SECTION_CLOSE

/* registration stuff */

VMDPLUGIN_API int VMDPLUGIN_init() {
  memset(&plugin, 0, sizeof(molfile_plugin_t));
  plugin.abiversion = vmdplugin_ABIVERSION;
  plugin.type = MOLFILE_PLUGIN_TYPE;
  plugin.name = "hoomd";
  plugin.prettyname = "HOOMD-blue XML File";
  plugin.author = "Axel Kohlmeyer";
  plugin.majorv = 0;
  plugin.minorv = 10;
  plugin.is_reentrant = VMDPLUGIN_THREADUNSAFE;
  plugin.filename_extension = "xml";

  plugin.open_file_read = open_hoomd_read;
  plugin.read_structure = read_hoomd_structure;
  plugin.read_bonds = read_hoomd_bonds;
  plugin.read_next_timestep = read_hoomd_timestep;
  plugin.close_file_read = close_hoomd_read;
#if vmdplugin_ABIVERSION > 10
  plugin.read_timestep_metadata    = read_timestep_metadata;
#endif

  plugin.open_file_write = open_hoomd_write;
  plugin.write_bonds = write_hoomd_bonds;
#if vmdplugin_ABIVERSION > 9
  plugin.read_angles = read_hoomd_angles;
  plugin.write_angles = write_hoomd_angles;
#endif
  plugin.write_structure = write_hoomd_structure;
  plugin.write_timestep = write_hoomd_timestep;
  plugin.close_file_write = close_hoomd_write;
  return VMDPLUGIN_SUCCESS;
}

VMDPLUGIN_API int VMDPLUGIN_register(void *v, vmdplugin_register_cb cb) {
  (*cb)(v, (vmdplugin_t *)&plugin);
  return VMDPLUGIN_SUCCESS;
}

VMDPLUGIN_API int VMDPLUGIN_fini() {
  return VMDPLUGIN_SUCCESS;
}


#ifdef TEST_PLUGIN

int main(int argc, char *argv[]) {
  molfile_timestep_t timestep;
  molfile_atom_t *atoms;
  void *v, *w;
  int i, natoms, nbonds, optflags;
#if vmdplugin_ABIVERSION > 10
  molfile_timestep_metadata_t ts_meta;
#endif
  int nbondtypes, nangletypes, ndihtypes, nimptypes;
  int nangles, ndihedrals, nimpropers, ncterms;
  int ctermrows, ctermcols;
  int *from, *to, *bondtype, *angles, *angletype, *dihedrals, *dihedraltype;
  int *impropers, *impropertype, *cterms;
#if vmdplugin_ABIVERSION < 16
  double *angleforces, *dihedralforces, *improperforces, *ctermforces;
#endif
  float *order;
  char **btnames, **atnames, **dtnames, **itnames;

  bondtype=NULL;
  btnames=NULL;
  nbondtypes=0;
  nangletypes=0;

  VMDPLUGIN_init();
  
  while (--argc) {
    ++argv;
    v = open_hoomd_read(*argv, "hoomd", &natoms);
    if (!v) {
      fprintf(stderr, "open_hoomd_read failed for file %s\n", *argv);
      return 1;
    }
    fprintf(stderr, "open_hoomd_read succeeded for file %s\n", *argv);
    fprintf(stderr, "number of atoms: %d\n", natoms);

    atoms=(molfile_atom_t *)malloc(natoms*sizeof(molfile_atom_t));
    if (read_hoomd_structure(v, &optflags, atoms) != MOLFILE_SUCCESS) return 1;
#if vmdplugin_ABIVERSION > 14
    read_hoomd_bonds(v, &nbonds, &from, &to, &order, &bondtype, &nbondtypes, &btnames);

    read_hoomd_angles(v, &nangles, &angles, &angletype, &nangletypes, &atnames, 
                      &ndihedrals, &dihedrals, &dihedraltype, &ndihtypes, &dtnames, 
                      &nimpropers, &impropers, &impropertype, &nimptypes, &itnames,
                      &ncterms, &cterms, &ctermcols, &ctermrows);
#else
    read_hoomd_bonds(v, &nbonds, &from, &to, &order);
#if vmdplugin_ABIVERSION > 9
    read_hoomd_angles(v, &nangles, &angles, &angleforces, &ndihedrals, 
                      &dihedrals, &dihedralforces, &nimpropers, &impropers, 
                      &improperforces, &ncterms, &cterms, &ctermcols, 
                      &ctermrows, &ctermforces);
#endif
#endif
    fprintf(stderr, "found: %d atoms, %d bonds, %d bondtypes, %d angles, "
            "%d angletypes\nfound: %d dihedrals, %d impropers, %d cterms\n", 
            natoms, nbonds, nbondtypes, nangles, nangletypes, ndihedrals,
            nimpropers, ncterms);

    fputs("ATOMS:\n", stderr);
    for(i=0; (i<20) && (i<natoms); ++i) {
      fprintf(stderr,"%05d: %s/%s %d\n",i+1,atoms[i].name, 
              atoms[i].type, atoms[i].atomicnumber);
    }

    fputs("BONDS:\n", stderr);
    if (nbonds > 0) {
      if (bondtype && nbondtypes > 0) {
        for(i=0; (i<20) && (i<nbonds);++i) {
          fprintf(stderr,"%05d: %s/%d %d %d\n", i+1, btnames[bondtype[i]], 
                  bondtype[i], from[i], to[i]);
        }
      } else {
        for(i=0;(i<20) && (i<nbonds);++i) {
          fprintf(stderr,"%05d: %d %d\n",i+1,from[i], to[i]);
        }
      }
    }
    
#if vmdplugin_ABIVERSION > 9
    fputs("ANGLES:\n", stderr);
    if (nangles > 0) {
      if (angletype && nangletypes > 0) {
        for(i=0; (i<20) && (i<nangles);++i) {
          fprintf(stderr,"%05d: %s/%d %d %d %d\n", i+1, atnames[angletype[i]], 
                  angletype[i], angles[3*i], angles[3*i+1], angles[3*i+2]);
        }
      } else {
        for(i=0;(i<20) && (i<nangles);++i) {
          fprintf(stderr,"%05d: %d %d %d\n",i+1, angles[3*i], 
                  angles[3*i+1], angles[3*i+2]);
        }
      }
    }
#endif
    
    i = 0;
    timestep.coords = (float *)malloc(3*sizeof(float)*natoms);
#if vmdplugin_ABIVERSION > 10
    /* XXX: this should be tested at each time step. */
    timestep.velocities = NULL;
    read_timestep_metadata(v,&ts_meta);
    if (ts_meta.has_velocities) {
      fprintf(stderr, "found timestep velocities metadata.\n");
    }
    timestep.velocities = (float *) malloc(3*natoms*sizeof(float));
#endif

    while (!read_hoomd_timestep(v, natoms, &timestep)) {
      i++;
    }
    fprintf(stderr, "ended read_next_timestep on frame %d\n", i);
    if (argc < 2) {
      w = open_hoomd_write("test.xml","hoomd",natoms);
#if vmdplugin_ABIVERSION > 14
      write_hoomd_bonds(w, nbonds, from, to, NULL, bondtype, nbondtypes, btnames);
#else
      write_hoomd_bonds(w, nbonds, from, to, NULL);
#endif
#if vmdplugin_ABIVERSION > 9
#if vmdplugin_ABIVERSION > 15
      write_hoomd_angles(w, nangles, angles, angletype, nangletypes, atnames, 
                         ndihedrals, dihedrals, dihedraltype, ndihtypes, dtnames, 
                         nimpropers, impropers, impropertype, nimptypes, itnames,
                         ncterms, cterms, ctermcols, ctermrows);
#else
      write_hoomd_angles(w, nangles, angles, NULL, ndihedrals, dihedrals, NULL,
                         nimpropers, impropers, NULL, ncterms, cterms, 
                         ctermcols, ctermrows, NULL);
#endif
#endif
      write_hoomd_structure(w, optflags, atoms);
      write_hoomd_timestep(w, &timestep);
      close_hoomd_write(w);
      fprintf(stderr, "done with writing hoomd file test.xml.\n");
    }
    close_hoomd_read(v);
    
    free(atoms);
    free(timestep.coords);
#if vmd_ABIVERSION > 10
    free(timestep.velocities);
#endif
  }
  VMDPLUGIN_fini();
  return 0;
}

#endif

