/* Code for simplifying the process of getting the right molfile plugin for a file
 * adapted/ripped off from VMDApp and catdcd, but reproduced separately to avoid 
 * depending on the VMD core
 */

#include "getplugins.h"
#include <string.h>

/* Data structures for storing all the molfile plugins and types */
static hash_t pluginhash; /* Stores the name/plugin associations */
static molfile_plugin_t *plugins[MAX_PLUGINS]; /* store pointers to the actual plugins */
static int num_plugins=0;

/* Register a plugin in our arrays */
/* Note that this code differs from the catdcd version -- allows multiple
 * extensions for each plugin */

static int register_plugin(void* v, vmdplugin_t* p) {  
  molfile_plugin_t* currplug = (molfile_plugin_t *)p;
  /* const char *key = p->name; */
  char* extbuf = strdup(currplug->filename_extension);
  char* extcur = extbuf;
  char* extnext = NULL;

  if (num_plugins >= MAX_PLUGINS) {
    fprintf(stderr, "Exceeded maximum allowed number of plugins; recompile. :(\n");
    return VMDPLUGIN_ERROR;
  }

/*  printf("Types for %s: %s\n", key, extbuf); */
  /* For each extension that the given plugin has... */
  while (extcur != NULL) {
    extnext = strchr(extcur, ',');
    if (extnext != NULL) {
      *extnext = '\0';
      extnext = &extnext[1];
    }

    /* Add that extension to the hash */
/*    printf("Adding plugin %s with index %i\n", extcur, num_plugins); */
    if (hash_insert(&pluginhash, strdup(extcur), num_plugins) != HASH_FAIL) {
/*      fprintf(stderr, "Multiple plugins for file type '%s' found!", key); */
      return VMDPLUGIN_ERROR;
    }

    extcur = extnext;

  }

  /* Add the current plugin to the plugin array */
  plugins[num_plugins] = currplug; 
  num_plugins += 1;

  free(extbuf);
  return VMDPLUGIN_SUCCESS;
}

/* Initialize our plugin lists and such */
void init_plugins() {
  hash_init(&pluginhash, 20);
  MOLFILE_INIT_ALL;
  MOLFILE_REGISTER_ALL(NULL, register_plugin);
}



/* look up a plugin for a given filetype */
molfile_plugin_t *get_plugin(const char* filetype) {
  int id;
  if ((id = hash_lookup(&pluginhash, filetype)) == HASH_FAIL) {
    fprintf(stderr, "No plugin found for filetype '%s'\n", filetype);
    return NULL;
  }
  return plugins[id];
}

/* look up the proper plugin for a file name */
molfile_plugin_t *get_plugin_forfile(const char* filename, char* filetype) {

  molfile_plugin_t *myplugin;

  /* Get the extension */
  
  char* ext = strrchr(filename, '.');
  if (!ext) {
    fprintf(stderr, "Couldn't find an extension in the filename '%s'\n", filename);
    return NULL;
  }

  ext = &ext[1];

  /* Make the extension uppercase and look it up in our hash table */
  myplugin = get_plugin(ext);

  strncpy(filetype, myplugin->name, 10);
  return myplugin;

}

     
