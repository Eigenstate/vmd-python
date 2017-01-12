/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: tngplugin.C,v $
 *      $Author: johns $       $Locker:  $             $State: Exp $
 *      $Revision: 1.7 $       $Date: 2015/10/12 03:00:46 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   VMD plugin to allow reading and writing of Gromacs TNG trajectories.
 *   This software and the TNG library are made available under the
 *   BSD license.  For details, see the license information associated
 *   with the Gromacs TNG library that this code depends on at:
 *     http://www.gromacs.org/
 *
 *   The TNG trajectory format is described in this publication:
 *     "An efficient and extensible format, library, and API for
 *     binary trajectory data from molecular simulations"
 *     Journal of Computational Chemistry 2013, DOI: 10.1002/jcc.23495
 *     http://onlinelibrary.wiley.com/doi/10.1002/jcc.23495/abstract
 *
 *   The TNG git repo is the master location for the sources, though
 *   the standard git SSL checks have to be disabled to access it:
 *      env GIT_SSL_NO_VERIFY=1 git clone https://gerrit.gromacs.org/tng
 *
 *   This version of the TNG plugin requires version 1.7.4 or greater
 *   of the TNG library.
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <tng/tng_io.h>
#include "molfile_plugin.h"

#define TNG_PLUGIN_MAJOR_VERSION 1
#define TNG_PLUGIN_MINOR_VERSION 0

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661922
#endif

typedef struct {
  tng_trajectory_t tng_traj;
  int natoms;
  int step;
//   int stride_length;
  int64_t n_frames;
  int coord_exponential;
//   int64_t n_frames_per_frame_set;
  double time_per_frame;
  int has_velocities;
} tngdata;

static void convert_tng_box_shape_to_vmd(float *box_shape, float *vmd_box)
{
    float A, B, C;

    A = sqrt(box_shape[0]*box_shape[0] +
             box_shape[1]*box_shape[1] +
             box_shape[2]*box_shape[2]);
    B = sqrt(box_shape[3]*box_shape[3] +
             box_shape[4]*box_shape[4] +
             box_shape[5]*box_shape[5]);
    C = sqrt(box_shape[6]*box_shape[6] +
             box_shape[7]*box_shape[7] +
             box_shape[8]*box_shape[8]);

    if ((A<=0) || (B<=0) || (C<=0))
    {
        vmd_box[0] = vmd_box[1] = vmd_box[2] = 0;
        vmd_box[3] = vmd_box[4] = vmd_box[5] = 90;
    }
    else
    {
        vmd_box[0] = A;
        vmd_box[1] = B;
        vmd_box[2] = C;
        vmd_box[3] = acos( (box_shape[3]*box_shape[6]+
                            box_shape[4]*box_shape[7]+
                            box_shape[5]*box_shape[8])/(B*C) ) * 90.0/M_PI_2;
        vmd_box[4] = acos( (box_shape[0]*box_shape[6]+
                            box_shape[1]*box_shape[7]+
                            box_shape[2]*box_shape[8])/(A*C) ) * 90.0/M_PI_2;
        vmd_box[5] = acos( (box_shape[0]*box_shape[3]+
                            box_shape[1]*box_shape[4]+
                            box_shape[2]*box_shape[5])/(A*B) ) * 90.0/M_PI_2;
    }
}

static void convert_vmd_box_shape_to_tng(const molfile_timestep_t *ts, float *box_shape)
{
//     const float sa = sin((double)ts->alpha/180.0*M_PI);
    const float ca = cos((double)ts->alpha/180.0*M_PI);
    const float cb = cos((double)ts->beta/180.0*M_PI);
    const float cg = cos((double)ts->gamma/180.0*M_PI);
    const float sg = sin((double)ts->gamma/180.0*M_PI);

    box_shape[0] = ts->A;
    box_shape[1] = 0.0;
    box_shape[2] = 0.0;
    box_shape[3] = ts->B*cg; // ts->B*ca when writing trr?!
    box_shape[4] = ts->B*sg; // ts->B*sa when writing trr?!
    box_shape[5] = 0.0;
    box_shape[6] = ts->C*cb;
    box_shape[7] = ts->C*(ca - cb*cg)/sg;
    box_shape[8] = ts->C*sqrt((double)(1.0 + 2.0*ca*cb*cg
                                       - ca*ca - cb*cb - cg*cg)/(1.0 - cg*cg));
}

static void *open_tng_read(const char *filename, const char*,
                           int *natoms)
{
    tngdata *tng;
    tng_function_status stat;
    int64_t n, exp;

    tng = new tngdata;

    stat = tng_util_trajectory_open(filename, 'r', &tng->tng_traj);
    if(stat != TNG_SUCCESS)
    {
        fprintf(stderr, "tngplugin) Cannot open file '%s'\n", filename);
        return NULL;
    }

    tng_num_particles_get(tng->tng_traj, &n);
    *natoms = (int)n;
    tng->natoms = (int)n;
    tng->step = 0;
    tng_num_frames_get(tng->tng_traj, &n);
    tng->n_frames = n;
    tng->has_velocities = 0;

    tng_distance_unit_exponential_get(tng->tng_traj, &exp);
    tng->coord_exponential = (int) exp;


    return tng;
}

static int read_tng_structure(void *v, int *optflags,
                               molfile_atom_t *atoms)
{
    tngdata *tng = (tngdata *)v;
    char long_name[16], short_name[2];
    int64_t id;

    *optflags = MOLFILE_NOOPTIONS;
    for(int i = 0; i < tng->natoms; i++)
    {
        molfile_atom_t *atom = atoms+i;
        tng_atom_name_of_particle_nr_get(tng->tng_traj, i, long_name, 16);
        strcpy(atom->name, long_name);
        tng_atom_type_of_particle_nr_get(tng->tng_traj, i, long_name, 16);
        strcpy(atom->type, long_name);
        tng_residue_name_of_particle_nr_get(tng->tng_traj, i, long_name, 16);
        strcpy(atom->resname, long_name);
        tng_global_residue_id_of_particle_nr_get(tng->tng_traj, i, &id);
        atom->resid = (int)id;
//         fprintf(stderr, "resid: %d\n", (int)id);
        tng_chain_name_of_particle_nr_get(tng->tng_traj, i, short_name, 2);
        strcpy(atom->chain, short_name);
        atom->segid[0] = '\0';
    }
//     fprintf(stderr, "Structure opened\n");
    return MOLFILE_SUCCESS;
}

#if vmdplugin_ABIVERSION > 14
static int read_tng_bonds(void *v, int *nbonds, int **fromptr, int **toptr,
                          float **bondorderptr, int **bondtypeptr,
                          int *nbondtypes, char ***bondtypename)
#else
static int read_tng_bonds(void *v, int *nbonds, int **fromptr, int **toptr,
                          float **bondorderptr)
#endif
{
    int64_t *from_atoms = 0, *to_atoms = 0, bond_cnt, i;
    tng_function_status stat;

    tngdata *tng = (tngdata *)v;

    stat = tng_molsystem_bonds_get(tng->tng_traj, &bond_cnt, &from_atoms,
                                   &to_atoms);
    if(stat != TNG_SUCCESS)
    {
        return MOLFILE_ERROR;
    }

    if(bond_cnt <= 0)
    {
        fprintf(stderr, "tngplugin) No bonds found in molsystem\n");
        *nbonds = 0;
        *fromptr = 0;
        *toptr = 0;
        return MOLFILE_SUCCESS;
    }

    /* Convert from int64_t to int. The fact that VMD and TNG use different
     * int types can lead to problems if there are very many atoms and/or
     * bonds */
    *nbonds = (int) bond_cnt;
    *fromptr = (int *)malloc((*nbonds) * sizeof(int));
    *toptr = (int *)malloc((*nbonds) * sizeof(int));
    *bondorderptr = 0;

#if vmdplugin_ABIVERSION > 14
    *bondtypeptr = 0;
    *nbondtypes = 0;
    *bondtypename = 0;
#endif

    for(i = 0; i < *nbonds; i++)
    {
        (*fromptr)[i] = (int)from_atoms[i] + 1;
        (*toptr)[i] = (int)to_atoms[i] + 1;
//         fprintf(stderr, "Adding bond from %d to %d\n", (*fromptr)[i], (*toptr)[i]);
    }

    return MOLFILE_SUCCESS;
}

static void convert_to_float(void *from, float *to, const float fact, const int natoms, const int nvalues, const char datatype)
{
    int i, j;

    switch(datatype)
    {
    case TNG_FLOAT_DATA:
        if(fact == 1)
        {
            memcpy(to, from, nvalues * sizeof(float) * natoms);
        }
        else
        {
            for(i = 0; i < natoms; i++)
            {
                for(j = 0; j < nvalues; j++)
                {
                    to[i*3+j] = (float)((float *)from)[i*3+j] * fact;
                }
            }
        }
        break;
    case TNG_INT_DATA:
        for(i = 0; i < natoms; i++)
        {
            for(j = 0; j < nvalues; j++)
            {
                to[i*3+j] = (float)((int64_t *)from)[i*3+j] * fact;
            }
        }
        break;
    case TNG_DOUBLE_DATA:
        for(i = 0; i < natoms; i++)
        {
            for(j = 0; j < nvalues; j++)
            {
                to[i*3+j] = (float)((double *)from)[i*3+j] * fact;
            }
        }
        break;
    default:
        fprintf(stderr, "tngplugin) Cannot cast data\n");
    }
    return;
}

static int read_tng_timestep(void *v, int natoms, molfile_timestep_t *ts)
{
    /* The pointers that will be allocated by the TNG must be NULL before allocation. */
    void *values = 0;
    char datatype;
    float box_shape[9], vmd_box[6];
    float fact = 1;
    int64_t frame, n, temp, temp2;
    tng_function_status stat;
    tngdata *tng = (tngdata *)v;

    if(!ts)
    {
        return MOLFILE_ERROR;
    }

//     fprintf(stderr, "Reading framestep from TNG\n");

    stat = tng_util_particle_data_next_frame_read(tng->tng_traj, TNG_TRAJ_POSITIONS, &values,
                                                  &datatype, &frame, &ts->physical_time);
    if(stat != TNG_SUCCESS)
    {
        return MOLFILE_ERROR;
    }
//     fprintf(stderr, "tngplugin) Timestep %d (%f), frame %d (%d), %d atoms\n",
//             tng->step, ts->physical_time, (int)frame, (int)tng->n_frames, natoms);

    /* TODO: Here it would be possible to add reading of the energy and pressure
     * measurements supported in VMD if they are present in the TNG file */
    tng_num_particles_get(tng->tng_traj, &n);
    if(n != natoms)
    {
        fprintf(stderr, "tngplugin) Timestep in file contains wrong number of atoms\n");
        fprintf(stderr, "tngplugin) Found %d, expected %d\n", (int)n, natoms);
        return MOLFILE_ERROR;
    }

    if(tng->coord_exponential != -10)
    {
        fact = pow(10.0, tng->coord_exponential + 10);
    }

    convert_to_float(values, ts->coords, fact, natoms, 3, datatype);

    if(ts->velocities)
    {
//         fprintf(stderr, "tngplugin) Reading velocities\n");
        stat = tng_particle_data_vector_interval_get(tng->tng_traj, TNG_TRAJ_VELOCITIES, frame,
                                                     frame, TNG_USE_HASH, &values,
                                                     &n, &temp, &temp2, &datatype);
        if(stat == TNG_CRITICAL)
        {
            return MOLFILE_ERROR;
        }
        if(stat == TNG_SUCCESS)
        {
            convert_to_float(values, ts->velocities, fact, natoms, 3, datatype);
        }
    }

    stat = tng_data_vector_interval_get(tng->tng_traj, TNG_TRAJ_BOX_SHAPE,
                                        frame, frame, TNG_USE_HASH, &values,
                                        &temp, &temp2, &datatype);
    if(stat == TNG_CRITICAL)
    {
        return MOLFILE_ERROR;
    }
    if(stat == TNG_SUCCESS)
    {
        convert_to_float(values, box_shape, fact, 1, 9, datatype);

        convert_tng_box_shape_to_vmd(box_shape, vmd_box);
        if(ts)
        {
            ts->A = vmd_box[0];
            ts->B = vmd_box[1];
            ts->C = vmd_box[2];
            ts->alpha = vmd_box[3];
            ts->beta = vmd_box[4];
            ts->gamma = vmd_box[5];
        }
    }

    ++tng->step;
    if(values)
    {
        free(values);
    }

    return MOLFILE_SUCCESS;
}

static int read_timestep_metadata(void *v, molfile_timestep_metadata_t *metadata)
{
    tng_function_status stat;
    tngdata *tng = (tngdata *)v;

    /* Check only once if there are velocities in the file at all. */
    if(tng->has_velocities == 0)
    {
        stat = tng_frame_set_read_current_only_data_from_block_id(tng->tng_traj, TNG_SKIP_HASH, TNG_TRAJ_VELOCITIES);

        if(stat == TNG_CRITICAL)
        {
            metadata->has_velocities = 0;
            return MOLFILE_ERROR;
        }
        else if(stat == TNG_SUCCESS)
        {
            fprintf(stderr, "tngplugin) Trajectory contains velocities\n");
            tng->has_velocities = 1;
        }
        else
        {
            fprintf(stderr, "tngplugin) Trajectory does not contain velocities\n");
            tng->has_velocities = -1;
        }
    }
    if(tng->has_velocities == 1)
    {
        metadata->has_velocities = 1;
    }
    else
    {
        metadata->has_velocities = 0;
    }

    return MOLFILE_SUCCESS;
}

static void close_tng(void *v)
{
    tngdata *tng = (tngdata *)v;
    tng_util_trajectory_close(&tng->tng_traj);
    delete tng;
}

static void *open_tng_write(const char *filename, const char*,
                            int natoms)
{
    tngdata *tng;
    tng_function_status stat;
//     int64_t n, exp;

    tng = new tngdata;

    stat = tng_util_trajectory_open(filename, 'w', &tng->tng_traj);
    if(stat != TNG_SUCCESS)
    {
        fprintf(stderr, "tngplugin) Cannot open file '%s'\n", filename);
        return NULL;
    }

    tng->natoms = natoms;
    tng->step = 0;
    tng->coord_exponential = -10;
    tng_distance_unit_exponential_set(tng->tng_traj, -10);

    tng->time_per_frame = -1;

    return tng;
}

static int write_tng_structure(void *v, int optflags, const molfile_atom_t *atoms)
{
    /* VMD atoms do not contain molecule information, which
     * complicates TNG writing a bit. */
    tng_molecule_t tng_mol = 0;
    tng_chain_t tng_chain = 0;
    tng_residue_t tng_residue = 0;
    tng_atom_t tng_atom;
    int prev_resid = -1;
    char new_chain_name[4] = "X_A";

    tngdata *tng = (tngdata *)v;

    /* A dummy molecule must be added. All atoms will be added to it. */
    if(tng_molecule_find(tng->tng_traj, "MOL", -1, &tng_mol) != TNG_SUCCESS)
    {
        tng_molecule_add(tng->tng_traj, "MOL", &tng_mol);
    }

    for(int i = 0; i < tng->natoms; i++)
    {
        if(tng_chain == 0)
        {
            if(tng_molecule_chain_find(tng->tng_traj, tng_mol, atoms[i].chain, -1, &tng_chain) !=
                TNG_SUCCESS)
            {
                tng_molecule_chain_add(tng->tng_traj, tng_mol, atoms[i].chain, &tng_chain);
            }
        }
        else if(tng_residue != 0)
        {
            if(prev_resid >= 0)
            {
                if(prev_resid > atoms[i].resid)
                {
                    new_chain_name[0] = atoms[i].chain[0];
                    ++new_chain_name[2];
                    tng_molecule_chain_add(tng->tng_traj, tng_mol, new_chain_name, &tng_chain);
                }
            }
        }
//         fprintf(stderr, "tngplugin) Looking for residue: %s, %d\n", atoms[i].resname, atoms[i].resid);
        if (tng_chain_residue_find(tng->tng_traj, tng_chain, atoms[i].resname,
                                   atoms[i].resid, &tng_residue) != TNG_SUCCESS)
        {
            tng_chain_residue_w_id_add(tng->tng_traj, tng_chain, atoms[i].resname,
                                       atoms[i].resid, &tng_residue);
            prev_resid = atoms[i].resid;
        }
        tng_residue_atom_add(tng->tng_traj, tng_residue, atoms[i].name, atoms[i].type, &tng_atom);
    }
    tng_molecule_cnt_set(tng->tng_traj, tng_mol, 1);

    return MOLFILE_SUCCESS;
}

#if vmdplugin_ABIVERSION > 14
static int write_tng_bonds(void *v, int nbonds, int *from, int *to,
                          float *bondorder, int *bondtype,
                          int nbondtypes, char **bondtypename)
#else
static int write_tng_bonds(void *v, int nbonds, int *from, int *to,
                          float *bondorder)
#endif
{
    tng_molecule_t tng_mol;
    tng_bond_t tng_bond;
    int i;

    if(nbonds == 0)
    {
        return MOLFILE_SUCCESS;
    }

    tngdata *tng = (tngdata *)v;

    /* A dummy molecule must be added. All atoms will be added to it. */
    if(tng_molecule_find(tng->tng_traj, "MOL", -1, &tng_mol) != TNG_SUCCESS)
    {
        tng_molecule_add(tng->tng_traj, "MOL", &tng_mol);
    }

    for (i = 0; i < nbonds; i++)
    {
        if(tng_molecule_bond_add(tng->tng_traj, tng_mol, from[i]-1, to[i]-1, &tng_bond) != TNG_SUCCESS)
        {
            fprintf(stderr, "tngplugin) Error adding bond %d (from %d to %d).\n", i, from[i], to[i]);
            return MOLFILE_ERROR;
        }
    }
    return MOLFILE_SUCCESS;
}

static int write_tng_timestep(void *v, const molfile_timestep_t *ts)
{
    float box_shape[9];
    tngdata *tng = (tngdata *)v;

    /* If there are fewer particles in the TNG mol system (write_tng_structure
     * has not already been performed) compensate by creating implicit particles,
     * which will not have full atom information. */
    tng_implicit_num_particles_set(tng->tng_traj, tng->natoms);

    if(!ts)
    {
        return MOLFILE_ERROR;
    }

    convert_vmd_box_shape_to_tng(ts, box_shape);
    if(tng->step == 1 && ts->physical_time != 0)
    {
        tng->time_per_frame = ts->physical_time;
        tng_time_per_frame_set(tng->tng_traj, tng->time_per_frame);
    }
    if(tng->time_per_frame < 0)
    {
//         fprintf(stderr, "tngplugin) Writing frame without time stamp\n");
        tng_util_box_shape_write(tng->tng_traj, tng->step, box_shape);
        tng_util_pos_write(tng->tng_traj, tng->step, ts->coords);
    }
    else
    {
//         fprintf(stderr, "tngplugin) Writing frame with time stamp\n");
        tng_util_box_shape_with_time_write(tng->tng_traj, tng->step, ts->physical_time,
                                           box_shape);
        tng_util_pos_with_time_write(tng->tng_traj, tng->step, ts->physical_time,
                                     ts->coords);
    }
    if(tng->step == 0)
    {
        tng_util_pos_write_interval_set(tng->tng_traj, 1);
        tng_util_box_shape_write_interval_set(tng->tng_traj, 1);
    }
    if(ts->velocities)
    {
//         fprintf(stderr, "tngplugin) Writing TNG velocities\n");
        if(tng->time_per_frame < 0)
        {
            tng_util_vel_write(tng->tng_traj, tng->step, ts->velocities);
        }
        else
        {
            tng_util_vel_with_time_write(tng->tng_traj, tng->step, ts->physical_time,
                                         ts->velocities);
        }
        if(tng->step == 0)
        {
            tng_util_vel_write_interval_set(tng->tng_traj, 1);
        }
    }

    tng->step++;

    return MOLFILE_SUCCESS;
}

static molfile_plugin_t tng_plugin;

VMDPLUGIN_API int VMDPLUGIN_init() {
  // TNG plugin init
  memset(&tng_plugin, 0, sizeof(molfile_plugin_t));
  tng_plugin.abiversion = vmdplugin_ABIVERSION;
  tng_plugin.type = MOLFILE_PLUGIN_TYPE;
  tng_plugin.name = "tng";
  tng_plugin.prettyname = "TNG: Trajectory Next Generation (testing)";
  tng_plugin.author = "Magnus Lundborg";
  tng_plugin.majorv = TNG_PLUGIN_MAJOR_VERSION;
  tng_plugin.minorv = TNG_PLUGIN_MINOR_VERSION;
  tng_plugin.is_reentrant = VMDPLUGIN_THREADUNSAFE;
  tng_plugin.filename_extension = "tng";
  tng_plugin.open_file_read = open_tng_read;
  tng_plugin.read_structure = read_tng_structure;
  tng_plugin.read_bonds = read_tng_bonds;
  tng_plugin.read_next_timestep = read_tng_timestep;
  tng_plugin.close_file_read = close_tng;
  tng_plugin.open_file_write = open_tng_write;
  tng_plugin.write_structure = write_tng_structure;
  tng_plugin.write_timestep = write_tng_timestep;
  tng_plugin.close_file_write = close_tng;
  tng_plugin.write_bonds = write_tng_bonds;
  tng_plugin.read_timestep_metadata = read_timestep_metadata;

  return VMDPLUGIN_SUCCESS;
}

VMDPLUGIN_API int VMDPLUGIN_register(void *v, vmdplugin_register_cb cb) {
  (*cb)(v, (vmdplugin_t *)&tng_plugin);
  return VMDPLUGIN_SUCCESS;
}

VMDPLUGIN_API int VMDPLUGIN_fini() {
  return VMDPLUGIN_SUCCESS;
}


