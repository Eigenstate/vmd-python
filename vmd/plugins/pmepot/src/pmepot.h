/*
 * $Id: pmepot.h,v 1.2 2005/07/20 15:37:39 johns Exp $
 *
 */

#ifndef PMEPOT_H
#define PMEPOT_H

typedef struct pmepot_data_struct pmepot_data;

pmepot_data* pmepot_create(int *dims, float ewald_factor);

void pmepot_destroy(pmepot_data *data);

int pmepot_add(pmepot_data *data, const float *cell,
		int natoms, const float *atoms);

int pmepot_writedx(pmepot_data *data, const char *filename);

#endif

