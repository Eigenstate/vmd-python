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
 *	$RCSfile: getplugins.h,v $
 *	$Author: ltrabuco $	$Locker:  $		$State: Exp $
 *	$Revision: 1.1 $	$Date: 2009/08/06 20:58:45 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Code for simplifying the process of getting the right molfile plugin 
 * for a file adapted/ripped off from catdcd
 *
 ***************************************************************************/

#ifndef _GETPLUGINS_H
#define _GETPLUGINS_H

#include "molfile_plugin.h"

/* Look up the plugin corresponding to a given file type */
/* Return a pointer to the right plugin */
molfile_plugin_t *get_plugin(const char*);

/* Look up the plugin for a given file NAME
 * For this purpose, we assume that the name is everything after the last .
 * and that the name corresponds to the plugin
 * As a side effect, the second argument is set to the extension */

molfile_plugin_t *get_plugin_forfile(const char*, char*);

/* List of all plugins that can read volumetric data */
char* plugins_read_volumetric_data();

/* List of all plugins that can write volumetric data */
char* plugins_write_volumetric_data();

#endif
