
/* 
 * molfile fortran interface
 * $Id: f77_molfile.c,v 1.1 2006/03/10 22:48:49 johns Exp $
 * (c) 2006 Axel Kohlmeyer <akohlmey@cmm.chem.upenn.edu>
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#include "molfile_plugin.h"
#include "libmolfile_plugin.h"
#include "vmdplugin.h"

#define F77TESTME 1

/* fortran name mangling */ 
#if defined(_F77_NO_UNDERSCORE)
#define FNAME(n) n
#elif defined(_F77_F2C_UNDERSCORE)
#define FNAME(n) n ## __
#else
#define FNAME(n) n ## _
#endif

/* interface typedef magic */
typedef int int4;

struct molfile_f77_handle 
{
    void  *handle;
    const char *fname;
    const char *ftype;
    molfile_plugin_t *plugin;
};
typedef struct molfile_f77_handle f77_fd;

/* plugin list */
#ifndef MAXPLUGINS
#define MAXPLUGINS 200
#endif
static int numplugins=0;
static molfile_plugin_t *plugin_list[MAXPLUGINS];

/* we maintain a static list of assigned handles */
#ifndef MAXHADNLES
#define MAXHANDLES 200
#endif
static int4 numhandles=0;
static f77_fd handle_list[MAXHANDLES];

/* helper functions */
/* helper function to copy fortran style (a la sun fortran) strings into
 * valid c style strings. just using the string pointers will not work,
 * since the strings are NOT zero terminated.
 *
 * WARNING: do not forget to free(2) them later, 
 * or you'll have a memory leak!
 */
static char *f77strdup(const char *s,const int sz)
{
    char *r;

    r = (char *)malloc(sz + 1);
    r = (char *)memcpy(r, s, sz);
    r[sz] = '\0';
    return r;
}

/* trim off whitespace at the end of a string */
static void f77trim(char *s,const int sz)
{
    int i;

    i=1;
    while( (i++ < sz) && isspace(s[sz-i]) ) {
        s[sz-i] ='\0';
    }
}

/* get the filename extension */
static const char *f77getfnext(const char *s)
{
    int i,len;

    len = strlen(s);
    for (i=len; i>=0; --i) {
        if(s[i] == '.') {
            return &s[i+1];
        }
    }
    return NULL;
}

/* check validity of plugins and register them. */
static int f77register(void *ptr, vmdplugin_t *plugin) {

    if (!plugin->type || !plugin->name || !plugin->author) {
        fprintf(stderr," skipping plugin with incomplete header\n");
        return -1;
    }

#if F77TESTME    
    fprintf(stderr, " trying to register plugin #%d: %s,  type:    %s/%d\n"
            " written by: %s\n\n", numplugins+1, plugin->name, 
            plugin->type, plugin->abiversion, plugin->author);
#endif

    if (plugin->abiversion != vmdplugin_ABIVERSION) {
        fprintf(stderr, " skipping plugin with incompatible ABI:%d/%d\n",
                plugin->abiversion, vmdplugin_ABIVERSION);
        return -2;
    }

    if (0 != strncmp(plugin->type, "mol file", 8)) {
        fprintf(stderr, " skipping plugin of incompatible type:%s\n",
                plugin->type);
        return -3;
    }

    if (numplugins < MAXPLUGINS) {
        plugin_list[numplugins] = (molfile_plugin_t *) plugin;
        ++numplugins;
        return 0;
    }
    
    fprintf(stderr, " too many plugins: %d. increase MAXPLUGINS, "
            "recompile, and try again.\n", numplugins);
    
    return -4;
}


/* the official fortran API */

/* register all available plugins and clear handles. */
void FNAME(f77_molfile_init)(void) 
{
    int i;
    
    MOLFILE_INIT_ALL;

    for (i=0; i<MAXHANDLES; ++i) {
        handle_list[i].handle = NULL;
    }

    MOLFILE_REGISTER_ALL(NULL,f77register);

    /* 
     * FIXME: check all plugins and make 
     * sure the babel plugin(s) are last.
     */
}

/* unregister all available plugins */
void FNAME(f77_molfile_finish)(void) 
{
#if 0
    int i;

    /* FIXME: add code to close and nullify all open handles */
    for (i=0; i<MAXHANDLES; ++i) {
        handle_list[i] = NULL;
    }
#endif

    MOLFILE_FINI_ALL;
}


/* open a file and provide file descriptor */
void FNAME(f77_molfile_open_read)(int4 *handle, int4 *natoms,
                        const char *infile, const char *intype, 
                        const int len_if, const int len_it)
{
    char *fname, *ftype;
    molfile_plugin_t *plugin;
    int i;
    
    if (numhandles >= MAXHANDLES) {
        fprintf(stderr, "too many molfile f77 handles.\n");
        *handle = -666;
        return;
    }

    fname = f77strdup(infile, len_if);
    f77trim(fname,len_if);
    
    ftype = f77strdup(intype, len_it);
    f77trim(ftype,len_it);
            
    fprintf(stderr, " %s: trying for: %s/%d, %s/%d\n", 
            __FUNCTION__, fname, len_if, ftype, len_it);

    plugin = NULL;
    /* determine plugin type automatically */
    if(0 == strncmp(intype, "auto", 4)) {
        const char *fext;
        
        fext = f77getfnext(fname);
        if (fext == NULL) {
            fprintf(stderr, " could not determine file name extension "
                    "for automatic plugin guess\n");
            *handle = -111;
            return;
        }
#if F77TESTME
        fprintf(stderr, " filename extension: %s\n", fext);
#endif

        for (i=0; (i<numplugins) && plugin==NULL; ++i) {
#if F77TESTME
            fprintf(stderr, " tying filename extension: %s\n",
                    plugin_list[i]->filename_extension);
#endif
            if (0 == strcmp(plugin_list[i]->filename_extension, fext)) {
                fprintf(stderr, " using plugin: %s\n", 
                        plugin_list[i]->prettyname);
                
                plugin = plugin_list[i];
            }
        }
        if (plugin == NULL) {
            fprintf(stderr, " could not determine matching plugin type"
                    "from file name extension\n");
            *handle = -222;
            return;
        }
    } else {
        
        for (i=0; (i<numplugins) && (plugin==NULL); ++i) {
#if F77TESTME
            fprintf(stderr, " tying plugin type: %s\n",
                    plugin_list[i]->name);
#endif
            if (0 == strcmp(plugin_list[i]->name, ftype)) {
                fprintf(stderr, " using plugin: %s\n", 
                        plugin_list[i]->prettyname);
                plugin = plugin_list[i];
            }
        }
        if (plugin == NULL) {
            fprintf(stderr, " could not find plugin for type %s\n",ftype);
            *handle = -333;
            return;
        }
    }
    
    if(plugin == NULL) { /* this should not happen, but... */
        fprintf(stderr, " no plugin found.\n");
        *handle = -444;
        return;
    }
    
    /* build handle */
    ++numhandles;
    for (i=0; i<numhandles; ++i) {
        if(handle_list[i].plugin == NULL) {
            *handle = i;
            handle_list[i].fname=fname;
            handle_list[i].ftype=plugin->name;
            handle_list[i].plugin=plugin;
        }
    }

    /* open file for reading and detect number of atoms */
    *natoms=MOLFILE_NUMATOMS_UNKNOWN;
    handle_list[*handle].handle= 
        plugin->open_file_read(fname,plugin->name,natoms);
    if(handle_list[*handle].handle == NULL) {
        fprintf(stderr, " open of %s-plugin for file %s failed\n",
                plugin->type, fname);
        --numhandles;
        handle_list[*handle].plugin=NULL;
        *handle=-777;
        return;
    }
    
    return;
}

/* read next time step */
void FNAME(f77_molfile_read_next)(int4 *handle, int4 *natoms, float *xyz, 
                             float *box, int4 *status)
{
    molfile_plugin_t *plugin;
    molfile_timestep_t step;
    int retval;

    /* do some sanity checks on the handle */
    if((*handle < 0) || (*handle >= MAXHANDLES)) {
        fprintf(stderr, " %s: illegal handle: %d\n",
                __FUNCTION__, *handle);
        *status = 0;
        return;
    }

    plugin = handle_list[*handle].plugin;
    if(plugin==NULL) {
        fprintf(stderr, " %s: inactive handle: %d\n",
                __FUNCTION__, *handle);
        *status = 0;
        return;
    }

    /* skip or read the timestep as demanded */
    if(status == 0) {
        retval = plugin->read_next_timestep(handle_list[*handle].handle,
                                             *natoms, NULL);
    } else {
        step.coords = xyz;
        retval = plugin->read_next_timestep(handle_list[*handle].handle,
                                             *natoms, &step);
    }

    /* copy the box parameters */
    if (retval == MOLFILE_SUCCESS) {
        *status = 1;
        box[0]=step.A;
        box[1]=step.B;
        box[2]=step.C;
        box[3]=step.alpha;
        box[4]=step.beta;
        box[5]=step.gamma;
    } else {
        *status = 0;
    }
}
            
/* close a read file descriptor */
void FNAME(f77_molfile_close_read)(int4 *handle)
{
    molfile_plugin_t *plugin;
    
    /* do some sanity checks on the handle */
    if((*handle < 0) || (*handle >= MAXHANDLES)) {
        fprintf(stderr, " %s: illegal handle: %d\n",
                __FUNCTION__, *handle);
        *handle = -111;
        return;
    }

    plugin = handle_list[*handle].plugin;
    if(plugin==NULL) {
        fprintf(stderr, " %s: inactive handle: %d\n",
                __FUNCTION__, *handle);
        *handle = -222;
        return;
    }

#if F77TESTME
    fprintf(stderr, " %s: trying to close handle %d"
            " for file %s\n", __FUNCTION__, *handle, 
            handle_list[*handle].fname);
#endif

    plugin->close_file_read(handle_list[*handle].handle);
    --numhandles;
    handle_list[*handle].plugin=NULL;
    *handle=-1;
}
