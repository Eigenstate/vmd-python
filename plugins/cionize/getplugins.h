/* Code for simplifying the process of getting the right molfile plugin for a file
 * adapted/ripped off from catdcd
 */

#ifndef CIONIZE_PLUGINS
#define CIONIZE_PLUGINS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hash.h"
#include "libmolfile_plugin.h"
#include "molfile_plugin.h"

#define MAX_PLUGINS 200


/* Initialize the hash and plugin arrays */
void init_plugins(void);

/* Look up the plugin corresponding to a given file type */
/* Return a pointer to the right plugin */
molfile_plugin_t *get_plugin(const char*);

/* Look up the plugin for a given file NAME
 * For this purpose, we assume that the name is everything after the last .
 * and that the name corresponds to the plugin
 * As a side effect, the second argument is set to the extension */

molfile_plugin_t *get_plugin_forfile(const char*, char*);

#endif
