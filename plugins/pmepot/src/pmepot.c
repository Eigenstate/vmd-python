/*
 * Top level PME potential routines
 *
 * $Id: pmepot.c,v 1.4 2005/07/20 15:37:39 johns Exp $
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

struct pmepot_data_struct {
  int dims[5];
  int grid_size;
  int max_dim;
  int fft_ntable;
  float ewald_factor;
  float oddd[12];
  int avg_count;
  float *avg_potential;
  float *fft_table;
  float *fft_work;
};

#include "pmepot.h"
#include "pub3dfft.h"

pmepot_data* pmepot_create(int *dims, float ewald_factor) {
  pmepot_data *data;
  int grid_size, max_dim;

  if ( dims[0] < 8 ) return 0;
  if ( dims[1] < 8 ) return 0;
  if ( dims[2] < 8 ) return 0;
  if ( dims[2] % 2 ) return 0;
  if ( ewald_factor <= 0. ) return 0;

  data = malloc(sizeof(pmepot_data));
  if ( ! data ) return 0;

  data->avg_count = 0;
  data->ewald_factor = ewald_factor;
  data->dims[0] = dims[0];
  data->dims[1] = dims[1];
  data->dims[2] = dims[2];
  data->dims[3] = dims[1];
  data->dims[4] = dims[2] + 2;
  grid_size = data->dims[0] * data->dims[3] * data->dims[4];
  data->grid_size = grid_size;
  max_dim = dims[0] > dims[1] ? dims[0] : dims[1];
  max_dim = max_dim > dims[2] ? max_dim : dims[2];
  data->max_dim = max_dim;
  data->fft_ntable = 4*max_dim+15;

  data->avg_potential = malloc(grid_size * sizeof(float));
  data->fft_table = malloc(3 * data->fft_ntable * sizeof(float));
  data->fft_work = malloc(2 * max_dim * sizeof(float));
  if ( ! data->avg_potential || ! data->fft_table || ! data->fft_work ) {
    if ( data->avg_potential) free(data->avg_potential);
    if ( data->fft_table) free(data->fft_table);
    if ( data->fft_work) free(data->fft_work);
    free(data);
    return 0;
  }

  pubd3di(dims[2], dims[1], dims[0], data->fft_table, data->fft_ntable);

  return data;
}

void pmepot_destroy(pmepot_data *data) {
  free(data->avg_potential);
  free(data->fft_table);
  free(data->fft_work);
  free(data);
}

void scale_grid(const int *dims, float *arr, const float factor) {
  int grid_size, i;
  grid_size = dims[0] * dims[3] * dims[4];
  for ( i=0; i<grid_size; ++i ) arr[i] *= factor;
}

void add_to_grid(const int *dims, float *avg, const float *arr) {
  int grid_size, i;
  grid_size = dims[0] * dims[3] * dims[4];
  for ( i=0; i<grid_size; ++i ) avg[i] += arr[i];
}

int fill_charges(const int *dims, const float *cell, int natoms,
		const float *xyzq, float *q_arr, float *rcell, float *oddd);

float compute_energy(float *q_arr, const float *cell, const float *rcell,
                        const int *dims, float ewald);

#define COLOUMB 332.0636
#define BOLTZMAN 0.001987191

int pmepot_add(pmepot_data *data, const float *cell,
		int natoms, const float *atoms) {
  float *q_arr;
  float rcell[12];

  if ( data->avg_count == 0 ) {
    q_arr = data->avg_potential;
  } else {
    q_arr = malloc(data->grid_size * sizeof(float));
    if ( ! q_arr ) return -1;
  }

  fill_charges(data->dims,cell,natoms,atoms,q_arr,rcell,data->oddd);

  pubdz3d(1, data->dims[2], data->dims[1], data->dims[0],
   q_arr, data->dims[4], data->dims[3],
   data->fft_table, data->fft_ntable, data->fft_work);

  compute_energy(q_arr, cell, rcell, data->dims, data->ewald_factor);

  pubzd3d(-1, data->dims[2], data->dims[1], data->dims[0],
   q_arr, data->dims[4], data->dims[3],
   data->fft_table, data->fft_ntable, data->fft_work);

  scale_grid(data->dims,q_arr,COLOUMB/(300.0*BOLTZMAN));

  if ( data->avg_count ) {
    add_to_grid(data->dims,data->avg_potential,q_arr);
    free(q_arr);
  }
  data->avg_count += 1;

  return 0;
}

int write_dx_grid(FILE *file, const int *dims, const float *oddd,
                const float *data, float scale, const char *label);

int pmepot_writedx(pmepot_data *data, const char *filename) {
  FILE *file;
  int rval;
  if ( ! data->avg_count ) return -1;
  file = fopen(filename,"w");
  if ( ! file ) return -2;
  rval = write_dx_grid(file,data->dims,data->oddd,
			data->avg_potential,1./data->avg_count,
			"PME potential (kT/e, T=300K)");
  fclose(file);
  return rval * 10;
}

