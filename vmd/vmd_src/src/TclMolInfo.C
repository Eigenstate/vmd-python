/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: TclMolInfo.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.97 $	$Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   This is a helper function for the 'molinfo # get ...' command in
 * TclCommands.h .  It is used to access the information known about
 * molecules.
 *
 ***************************************************************************/

#include <stdlib.h> 
#include "MoleculeList.h"
#include "tcl.h"
#include "Timestep.h"
#include "TclCommands.h" // for my own, external function definitions
#include "Molecule.h"
#include "MaterialList.h"
#include "VMDApp.h"
#include "Inform.h"
#include "QMData.h"

/// XXX old-style context structure; from when we used a SymbolTable instance
/// to loop up functions for molinfo.
class ctxt {
public:
  int molinfo_get_index;
  int molinfo_get_id;
  int molinfo_frame_number;
  const char *molinfo_get_string;
  MoleculeList *mlist;
  MaterialList *matlist;
  VMDApp *app;
 
  ctxt() {
    molinfo_get_index = -1;
    molinfo_get_id = -1;
    molinfo_frame_number = -999;
    molinfo_get_string = NULL;
    mlist = NULL;
    matlist = NULL;
    app = NULL;
  }
};

static void write_matrix(Tcl_Interp *interp, const float *mat) {
  char s[16*25];
  sprintf(s, "{%g %g %g %g} {%g %g %g %g} {%g %g %g %g} {%g %g %g %g}", 
          mat[0], mat[4],  mat[8], mat[12],                 
          mat[1], mat[5],  mat[9], mat[13],                 
          mat[2], mat[6], mat[10], mat[14],                
          mat[3], mat[7], mat[11], mat[15]);                
  Tcl_AppendElement(interp, s);
}

static int read_matrix(Tcl_Interp *interp, const char *s, float *mat) {
  if (sscanf(s, 
        " { %f %f %f %f } { %f %f %f %f } { %f %f %f %f } { %f %f %f %f }",
        mat+0, mat+4, mat+ 8, mat+12,
        mat+1, mat+5, mat+ 9, mat+13,
        mat+2, mat+6, mat+10, mat+14,
        mat+3, mat+7, mat+11, mat+15) != 16) {
    Tcl_AppendResult(interp, "Matrix must contain 16 elements", NULL);
    return 0;
  }
  return 1;
}

#define generic_molinfo_data(name, func) \
} else if (!strcmp(arg, name)) { \
char buf[20]; sprintf(buf, "%d", func); Tcl_AppendElement(interp, buf);

#define generic_molinfo_simulation(name, term)  \
} else if (!strcmp(arg, name)) { \
Timestep *ts = mol->get_frame(context.molinfo_frame_number); \
if (!ts) Tcl_AppendElement(interp, "0"); \
else { char buf[20]; sprintf(buf, "%f", ts->energy[term]); Tcl_AppendElement(interp, buf); } 

#define generic_molinfo_pbc(name, term) \
} else if (!strcmp(arg, name)) { \
Timestep *ts = mol->get_frame(context.molinfo_frame_number); \
if (!ts) Tcl_AppendElement(interp, "0"); \
else { char buf[20]; sprintf(buf, "%f", ts->term); Tcl_AppendElement(interp, buf); }


#define generic_molinfo_wave_int(name, term) \
} else if (!strcmp(arg, name)) { \
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);  \
  Timestep *ts = mol->get_frame(context.molinfo_frame_number); \
  if (ts && ts->qm_timestep) { \
    int i; \
    char buf[32]; \
    for (i=0; i<ts->qm_timestep->get_num_wavef(); i++) { \
      sprintf(buf, "%d", ts->qm_timestep->get_##term(i)); \
      Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj(buf, -1)); \
    } \
  } \
  Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);

#define generic_molinfo_qmts_int(name, term) \
} else if (!strcmp(arg, name)) { \
  Timestep *ts = mol->get_frame(context.molinfo_frame_number); \
  if (!ts || !ts->qm_timestep) \
    Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), \
                             Tcl_NewIntObj(0)); \
  else { \
    char buf[20]; sprintf(buf, "%i", ts->qm_timestep->get_##term()); \
    Tcl_AppendElement(interp, buf); \
  }

#define generic_molinfo_qmts_arr(type, name, term, n) \
} else if (!strcmp(arg, name)) { \
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);  \
  Timestep *ts = mol->get_frame(context.molinfo_frame_number); \
  if (ts && ts->qm_timestep && ts->qm_timestep->get_##term()) { \
    char buf[20]; \
    unsigned int i; \
    for (i=0; i<(unsigned int)n; i++) { \
      sprintf(buf, type, ts->qm_timestep->get_##term()[i]); \
      Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj(buf, -1)); \
    } \
  } \
  Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);

#define generic_molinfo_qmts_mat(type, name, term, n, m)  \
} else if (!strcmp(arg, name)) { \
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);  \
  Timestep *ts = mol->get_frame(context.molinfo_frame_number); \
  if (ts && ts->qm_timestep && ts->qm_timestep->term) {             \
    char buf[20];  \
    unsigned int i, j;  \
    for (i=0; i<(unsigned int)n; i++) {  \
      Tcl_Obj *rowListObj = Tcl_NewListObj(0, NULL);  \
      for (j=0; j<(unsigned int)m; j++) {  \
        sprintf(buf, type, ts->qm_timestep->term[i*n+j]);  \
        Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewStringObj(buf, -1)); \
      }  \
      Tcl_ListObjAppendElement(interp, tcl_result, rowListObj);  \
    }  \
  } \
  Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);


#define generic_molinfo_qm(type, name, term) \
} else if (!strcmp(arg, name)) {  \
  if (!mol->qm_data) \
    Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), \
                             Tcl_NewIntObj(0)); \
  else {  \
    char buf[QMDATA_BUFSIZ]; sprintf(buf, type, mol->qm_data->term);  \
    Tcl_AppendElement(interp, buf);  \
  }

#define generic_molinfo_qm_string(name, term) \
} else if (!strcmp(arg, name)) {  \
  if (mol->qm_data) {  \
    char buf[QMDATA_BUFSIZ]; sprintf(buf, "%s", mol->qm_data->term);  \
    Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), Tcl_NewStringObj(buf, -1)); \
  } else { \
    Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), Tcl_NewListObj(0, NULL)); \
  }

#define generic_molinfo_qm_arr(type, name, term, n)  \
} else if (!strcmp(arg, name)) {  \
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);  \
  if (mol->qm_data && mol->qm_data->get_##term()) { \
    char buf[20]; \
    unsigned int i; \
    for (i=0; i<(unsigned int)n; i++) {  \
      sprintf(buf, type, mol->qm_data->get_##term()[i]); \
      Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj(buf, -1)); \
    }  \
  } \
  Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);

#define generic_molinfo_qm_mat(type, name, term, n)  \
} else if (!strcmp(arg, name)) {  \
  if (mol->qm_data) {  \
    if (mol->qm_data->get_##term()) {  \
      char buf[20];  \
      unsigned int i, j;  \
      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);  \
      for (i=0; i<(unsigned int)n; i++) {  \
        Tcl_Obj *rowListObj = Tcl_NewListObj(0, NULL);  \
        for (j=0; j<(unsigned int)n; j++) {  \
          sprintf(buf, type, mol->qm_data->get_##term()[i*n+j]);  \
          Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewStringObj(buf, -1)); \
        }  \
        Tcl_ListObjAppendElement(interp, tcl_result, rowListObj);  \
      }  \
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result); \
    }  \
  }


// given the list of strings, return the data
static int molinfo_get(ctxt context, int molid, int argc, const char *argv[],
                       Tcl_Interp *interp, int frame_num) {
  // does the molecule exist?
  context.molinfo_get_id = molid;
  context.molinfo_get_index = context.mlist->mol_index_from_id(molid);
  if (context.molinfo_get_index == -1) {
    char s[16];
    sprintf(s, "%d", molid);
    Tcl_AppendResult(interp, "molinfo: get: no molecule exists with id ",
                     s, NULL);
    return TCL_ERROR;
  }
  Molecule *mol = context.mlist->molecule(context.molinfo_get_index);

  // get the right frame number
  switch (frame_num) {
    case AtomSel::TS_NOW: 
      context.molinfo_frame_number = mol->frame();
      break;
    case AtomSel::TS_LAST:
      context.molinfo_frame_number = mol->numframes()-1;
      break;
    default:      
      context.molinfo_frame_number = frame_num;
  }

  for (int term=0; term<argc; term++) {
    context.molinfo_get_string = argv[term];

    // skip initial spaces
    const char *arg = argv[term];
    while (*arg == ' ')
      arg++;

    if (!strcmp(arg, "filename")) {
      Tcl_Obj *files = Tcl_NewListObj(0, NULL);
      for (int i=0; i<mol->num_files(); i++) {
        Tcl_ListObjAppendElement(interp, files, 
              Tcl_NewStringObj((char *)mol->get_file(i), -1));
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), files);
    } else if (!strcmp(arg, "index")) {
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), 
                               Tcl_NewIntObj(context.molinfo_get_index));
    } else if (!strcmp(arg, "id")) {
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), 
                               Tcl_NewIntObj(context.molinfo_get_id));
    } else if (!strcmp(arg, "numatoms")) {
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), 
                               Tcl_NewIntObj(mol->nAtoms));
    } else if (!strcmp(arg, "name")) {
      Tcl_AppendElement(interp, mol->molname());
    } else if (!strcmp(arg, "numreps")) {
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), 
                               Tcl_NewIntObj(mol->components()));
    } else if (!strcmp(arg, "numframes")) {
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), 
                               Tcl_NewIntObj(mol->numframes()));
    } else if (!strcmp(arg, "numvolumedata")) {
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), 
                               Tcl_NewIntObj(mol->num_volume_data()));
    } else if (!strcmp(arg, "last")) {
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), 
                               Tcl_NewIntObj(mol->numframes()-1));
    } else if (!strcmp(arg, "frame")) {
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), 
                               Tcl_NewIntObj(mol->frame()));
    } else if (!strcmp(arg, "filespec")) {
      Tcl_Obj *specs = Tcl_NewListObj(0, NULL);
      for (int i=0; i<mol->num_files(); i++) {
        Tcl_ListObjAppendElement(interp, specs, 
            Tcl_NewStringObj((char *)mol->get_file_specs(i), -1));
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), specs);
    } else if (!strcmp(arg, "filetype")) {
      Tcl_Obj *types = Tcl_NewListObj(0, NULL);
      for (int i=0; i<mol->num_files(); i++) {
        Tcl_ListObjAppendElement(interp, types, 
            Tcl_NewStringObj((char *)mol->get_type(i), -1));
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), types);
    } else if (!strcmp(arg, "database")) {
      Tcl_Obj *dbs = Tcl_NewListObj(0, NULL);
      for (int i=0; i<mol->num_files(); i++) {
        Tcl_ListObjAppendElement(interp, dbs, 
            Tcl_NewStringObj((char *)mol->get_database(i), -1));
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), dbs);
    } else if (!strcmp(arg, "accession")) {
      Tcl_Obj *acs = Tcl_NewListObj(0, NULL);
      for (int i=0; i<mol->num_files(); i++) {
        Tcl_ListObjAppendElement(interp, acs, 
            Tcl_NewStringObj((char *)mol->get_accession(i), -1));
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), acs);
    } else if (!strcmp(arg, "remarks")) {
      Tcl_Obj *rmk = Tcl_NewListObj(0, NULL);
      for (int i=0; i<mol->num_files(); i++) {
        Tcl_ListObjAppendElement(interp, rmk, 
            Tcl_NewStringObj((char *)mol->get_remarks(i), -1));
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), rmk);
    } else if (!strcmp(arg, "center")) {
      char s[50];
      sprintf(s, "%f %f %f", -mol->centt[0], -mol->centt[1], -mol->centt[2]);
      Tcl_AppendElement(interp, s);
    } else if (!strcmp(arg, "center_matrix")) {
      Matrix4 m;
      m.translate(mol->centt);
      write_matrix(interp, m.mat);
    } else if (!strcmp(arg, "rotate_matrix")) {
      write_matrix(interp, mol->rotm.mat);
    } else if (!strcmp(arg, "scale_matrix")) {
      Matrix4 m;
      m.scale(mol->scale);
      write_matrix(interp, m.mat);
    } else if (!strcmp(arg, "global_matrix")) {
      Matrix4 m;
      m.translate(mol->globt);
      write_matrix(interp, m.mat);
    } else if (!strcmp(arg, "view_matrix")) {
      write_matrix(interp, mol->tm.mat);
    } else if (!strncmp(arg, "rep", 3)) {
      const char *tmp = arg;
      while (*tmp++ != ' ');
      while (*tmp++ == ' ');
      int repnum = atoi(tmp-1);
      if (repnum<0 || repnum >= mol->components()) {
        Tcl_AppendResult(interp, arg, " out of range", NULL);
        return TCL_ERROR;
      }
      Tcl_AppendElement(interp, mol->component(repnum)->atomRep->cmdStr);
    } else if (!strncmp(arg, "selection", 9)) {
      const char *tmp = arg;
      while (*tmp++ != ' ');
      while (*tmp++ == ' ');
      int repnum = atoi(tmp-1);
      if (repnum<0 || repnum >= mol->components()) {
        Tcl_AppendResult(interp, arg, " out of range", NULL);
        return TCL_ERROR;
      }
      Tcl_AppendElement(interp, mol->component(repnum)->atomSel->cmdStr);
    } else if (!strncmp(arg, "color", 5)) {
      const char *tmp = arg;
      while (*tmp++ != ' ');
      while (*tmp++ == ' ');
      int repnum = atoi(tmp-1);
      if (repnum >= mol->components()) {
        Tcl_AppendResult(interp, arg, " out of range", NULL);
        return TCL_ERROR;
      }
      Tcl_AppendElement(interp, mol->component(repnum)->atomColor->cmdStr);
    } else if (!strncmp(arg, "material", 8)) {
      const char *tmp = arg;
      while (*tmp++ != ' ');
      while (*tmp++ == ' ');
      int repnum = atoi(tmp-1);
      if (repnum >= mol->components()) {
        Tcl_AppendResult(interp, arg, " out of range", NULL);
        return TCL_ERROR;
      }
      int matind = mol->component(repnum)->curr_material();
      Tcl_AppendElement(interp, context.matlist->material_name(matind));

    generic_molinfo_data("active", mol->active)
    generic_molinfo_data("drawn", mol->displayed())
    generic_molinfo_data("displayed", mol->displayed())
    generic_molinfo_data("fixed", mol->fixed())
    generic_molinfo_data("top", context.mlist->is_top(context.molinfo_get_index))
    generic_molinfo_simulation("bond", TSE_BOND)
    generic_molinfo_simulation("angle", TSE_ANGLE)
    generic_molinfo_simulation("dihedral", TSE_DIHE)
    generic_molinfo_simulation("improper", TSE_IMPR)
    generic_molinfo_simulation("vdw", TSE_VDW)
    generic_molinfo_simulation("electrostatic", TSE_COUL)
    generic_molinfo_simulation("elec", TSE_COUL)
    generic_molinfo_simulation("hbond", TSE_HBOND)
    generic_molinfo_simulation("kinetic", TSE_KE)
    generic_molinfo_simulation("potential", TSE_PE)
    generic_molinfo_simulation("temperature", TSE_TEMP)
    generic_molinfo_simulation("temp", TSE_TEMP)
    generic_molinfo_simulation("energy", TSE_TOTAL)
    generic_molinfo_simulation("volume", TSE_VOLUME)
    generic_molinfo_simulation("pressure", TSE_PRESSURE)
    generic_molinfo_simulation("efield", TSE_EFIELD)
    generic_molinfo_simulation("urey_bradley", TSE_UREY_BRADLEY)
    generic_molinfo_simulation("molinfo_restraint", TSE_RESTRAINT)

    } else if (!strcmp(arg, "timesteps")) {
      Timestep *ts = mol->get_frame(context.molinfo_frame_number);
      if (!ts) {
        Tcl_AppendElement(interp, "0");
      } else { 
        char buf[20]; 
        sprintf(buf, "%d", ts->timesteps);
        Tcl_AppendElement(interp, buf);
      }

    generic_molinfo_pbc("a", a_length)
    generic_molinfo_pbc("b", b_length)
    generic_molinfo_pbc("c", c_length)
    generic_molinfo_pbc("alpha", alpha)
    generic_molinfo_pbc("beta", beta)
    generic_molinfo_pbc("gamma", gamma)
    generic_molinfo_pbc("physical_time", physical_time)

    // Orbital info
    generic_molinfo_qmts_int("numscfiter",  num_scfiter)
    generic_molinfo_qmts_int("numwavef", num_wavef);
    generic_molinfo_wave_int("numorbitals", num_orbitals);
    generic_molinfo_wave_int("multiplicity", multiplicity);
    generic_molinfo_qmts_arr("%.12f", "scfenergy", scfenergies, ts->qm_timestep->get_num_scfiter());
    generic_molinfo_qmts_arr("%.6f", "gradients", gradients, 3L*mol->nAtoms);

    generic_molinfo_qm("%d", "nimag", get_num_imag());
    generic_molinfo_qm("%d", "nintcoords", get_num_intcoords());
//generic_molinfo_qm("%d", "ncart", get_num_cartcoords());
    generic_molinfo_qm("%d", "numbasis", num_basis);
    generic_molinfo_qm("%d", "numshells", get_num_shells());

    generic_molinfo_qm("%d", "num_basis_funcs", num_wave_f);
    generic_molinfo_qm("%d", "numelectrons", num_electrons);
    generic_molinfo_qm("%d", "totalcharge",  totalcharge);
    generic_molinfo_qm("%d", "nproc",   nproc);
    generic_molinfo_qm("%d", "memory", memory);
    generic_molinfo_qm_string("runtype", get_runtype_string());
    generic_molinfo_qm_string("scftype", get_scftype_string());
    generic_molinfo_qm_string("basis_string", basis_string);
    generic_molinfo_qm_string("runtitle", runtitle);
    generic_molinfo_qm_string("geometry", geometry);
    generic_molinfo_qm_string("qmversion", version_string);
    generic_molinfo_qm_string("qmstatus", get_status_string());

    generic_molinfo_qm_arr("%d", "num_shells_per_atom", num_shells_per_atom, mol->nAtoms);
    generic_molinfo_qm_arr("%d", "num_prim_per_shell", num_prim_per_shell, mol->qm_data->get_num_shells());
//    generic_molinfo_qm_arr("%d", "shell_types", shell_types, mol->qm_data->num_basis);
//    generic_molinfo_qm_arr("%d", "angular_momentum", angular_momentum, mol->qm_data->num_wave_f);
//    generic_molinfo_qm_arr("%.6f", "basis_array", basis_array, 2L*mol->qm_data->num_basis);

    } else if (!strcmp(arg, "basis")) {
      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
      if (mol->qm_data && mol->qm_data->get_basis()) {             
        char buf[20];                       
        int i, j, k;
        for (i=0; i<mol->nAtoms; i++) {
          if (!mol->qm_data->get_basis(i)) break;
          Tcl_Obj *atomListObj = Tcl_NewListObj(0, NULL);
          for (j=0; j<mol->qm_data->get_basis(i)->numshells; j++) {
            const shell_t *shell = mol->qm_data->get_basis(i, j);
            Tcl_Obj *shellListObj = Tcl_NewListObj(0, NULL);
            Tcl_Obj *exponListObj = Tcl_NewListObj(0, NULL);
            Tcl_Obj *coeffListObj = Tcl_NewListObj(0, NULL);
            for (k=0; k<shell->numprims; k++) {
              sprintf(buf, "%.7f", shell->prim[k].expon);  
              Tcl_ListObjAppendElement(interp, exponListObj, Tcl_NewStringObj(buf, -1));
              sprintf(buf, "%.12f", shell->prim[k].coeff);  
              Tcl_ListObjAppendElement(interp, coeffListObj, Tcl_NewStringObj(buf, -1));
            }
            sprintf(buf, "%s", mol->qm_data->get_shell_type_str(shell));
            Tcl_ListObjAppendElement(interp, shellListObj, Tcl_NewStringObj(buf, -1));
            Tcl_ListObjAppendElement(interp, shellListObj, exponListObj);
            Tcl_ListObjAppendElement(interp, shellListObj, coeffListObj);
            Tcl_ListObjAppendElement(interp, atomListObj, shellListObj);
          }
          Tcl_ListObjAppendElement(interp, tcl_result, atomListObj);
        }
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);

    } else if (!strcmp(arg, "overlap_matrix")) {
      Timestep *ts = mol->get_frame(context.molinfo_frame_number);
      if (ts && ts->qm_timestep &&
          mol->qm_data && mol->qm_data->get_basis()) {
        Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);

        float *overlap_matrix = NULL;
        float *expandedbasis = NULL;
        int *numprims = NULL;
        mol->qm_data->expand_basis_array(expandedbasis, numprims);
        mol->qm_data->compute_overlap_integrals(ts, expandedbasis,
                                                numprims,
                                                overlap_matrix);
        delete [] expandedbasis;
        delete [] numprims;
        int i, j;
        int numwavef = mol->qm_data->num_wave_f;
        char buf[20];
        for (i=0; i<numwavef; i++) {
          for (j=i; j<numwavef; j++) {
             sprintf(buf, "%.6f", overlap_matrix[i*numwavef+j]);
             Tcl_ListObjAppendElement(interp, tcl_result,
                 Tcl_NewStringObj(buf, -1));

           }
        }
        Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);
        delete [] overlap_matrix;
      }

    } else if (!strcmp(arg, "homo")) {
      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
      Timestep *ts = mol->get_frame(context.molinfo_frame_number);
      if (ts && ts->qm_timestep && ts->qm_timestep->get_num_wavef() &&
          mol->qm_data) {
        int iwave;
        for (iwave=0; iwave<ts->qm_timestep->get_num_wavef(); iwave++) {
          Tcl_ListObjAppendElement(interp, tcl_result,
             Tcl_NewIntObj(ts->qm_timestep->get_homo(iwave)));
        }
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);

    } else if (!strcmp(arg, "numavailorbs")) {
      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
      if (mol->qm_data) {
        int i;
        for (i=0; i<mol->qm_data->num_wavef_signa; i++) {
          Tcl_ListObjAppendElement(interp, tcl_result,
             Tcl_NewIntObj(mol->qm_data->get_max_avail_orbitals(i)));
        }
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);

    } else if (!strcmp(arg, "wavef_type")) {
      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
      Timestep *ts = mol->get_frame(context.molinfo_frame_number);
      if (ts && ts->qm_timestep && ts->qm_timestep->get_num_wavef() &&
          mol->qm_data) {
        int iwave;
        for (iwave=0; iwave<ts->qm_timestep->get_num_wavef(); iwave++) {
          char *buf;
          ts->qm_timestep->get_wavef_typestr(iwave, buf);
          Tcl_ListObjAppendElement(interp, tcl_result,
                     Tcl_NewStringObj(buf, -1));
          delete [] buf;
        }
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);

    } else if (!strcmp(arg, "wavef_spin")) {
      Timestep *ts = mol->get_frame(context.molinfo_frame_number);
      if (ts && ts->qm_timestep && ts->qm_timestep->get_num_wavef() &&
          mol->qm_data) {
        Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
        int iwave;
        for (iwave=0; iwave<ts->qm_timestep->get_num_wavef(); iwave++) {
          Tcl_ListObjAppendElement(interp, tcl_result,
                 Tcl_NewIntObj(ts->qm_timestep->get_spin(iwave)));
        }
        Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);
      }

    } else if (!strcmp(arg, "wavef_excitation")) {
      Timestep *ts = mol->get_frame(context.molinfo_frame_number);
      if (ts && ts->qm_timestep && ts->qm_timestep->get_num_wavef() &&
          mol->qm_data) {
        Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
        int iwave;
        for (iwave=0; iwave<ts->qm_timestep->get_num_wavef(); iwave++) {
          Tcl_ListObjAppendElement(interp, tcl_result,
                 Tcl_NewIntObj(ts->qm_timestep->get_excitation(iwave)));
        }
        Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);
      }

    } else if (!strcmp(arg, "wavef_energy")) {
      Timestep *ts = mol->get_frame(context.molinfo_frame_number);
      if (ts && ts->qm_timestep && ts->qm_timestep->get_num_wavef() &&
          mol->qm_data) {
        Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
        int iwave;
        for (iwave=0; iwave<ts->qm_timestep->get_num_wavef(); iwave++) {
          Tcl_ListObjAppendElement(interp, tcl_result,
                 Tcl_NewDoubleObj(ts->qm_timestep->get_wave_energy(iwave)));
        }
        Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);
      }

    } else if (!strcmp(arg, "orbenergies")) {
      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
      Timestep *ts = mol->get_frame(context.molinfo_frame_number);
      if (ts && ts->qm_timestep && ts->qm_timestep->get_num_wavef() &&
          mol->qm_data) {
        char buf[20];                       
        int iwave, orb;
        for (iwave=0; iwave<ts->qm_timestep->get_num_wavef(); iwave++) {
          Tcl_Obj *waveListObj = Tcl_NewListObj(0, NULL);
          const float *orben = ts->qm_timestep->get_orbitalenergy(iwave);
          if (orben) {
            int norbitals = ts->qm_timestep->get_num_orbitals(iwave);
            for (orb=0; orb<norbitals; orb++) {
              sprintf(buf, "%.12f", orben[orb]);
              Tcl_ListObjAppendElement(interp, waveListObj, Tcl_NewStringObj(buf, -1));
            }
          }
          Tcl_ListObjAppendElement(interp, tcl_result, waveListObj);
        }
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);

    } else if (!strcmp(arg, "orboccupancies")) {
      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
      Timestep *ts = mol->get_frame(context.molinfo_frame_number);
      if (ts && ts->qm_timestep && ts->qm_timestep->get_num_wavef() &&
          mol->qm_data) {
        char buf[20];                       
        int iwave, orb;
        for (iwave=0; iwave<ts->qm_timestep->get_num_wavef(); iwave++) {
          Tcl_Obj *waveListObj = Tcl_NewListObj(0, NULL);
          const float *occ = ts->qm_timestep->get_occupancies(iwave);
          int norbitals = ts->qm_timestep->get_num_orbitals(iwave);
          for (orb=0; orb<norbitals; orb++) {
            sprintf(buf, "%.12f", occ[orb]);
            Tcl_ListObjAppendElement(interp, waveListObj, Tcl_NewStringObj(buf, -1));
          }
          Tcl_ListObjAppendElement(interp, tcl_result, waveListObj);
        }
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);

    } else if (!strcmp(arg, "wavefunction")) {
      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
      Timestep *ts = mol->get_frame(context.molinfo_frame_number);
      if (ts && ts->qm_timestep && ts->qm_timestep->get_num_wavef() &&
          mol->qm_data) {
        char buf[20];                       
        int iwave, orb, i;
        for (iwave=0; iwave<ts->qm_timestep->get_num_wavef(); iwave++) {
          Tcl_Obj *waveListObj = Tcl_NewListObj(0, NULL);
          int norbitals = ts->qm_timestep->get_num_orbitals(iwave);
          for (orb=0; orb<norbitals; orb++) {
            Tcl_Obj *orbListObj = Tcl_NewListObj(0, NULL);
            const float *wave_f = ts->qm_timestep->get_wavecoeffs(iwave) + 
                                  orb*norbitals;
            for (i=0; i<ts->qm_timestep->get_num_coeffs(iwave); i++) {
              sprintf(buf, "%.12f", wave_f[i]);
              Tcl_ListObjAppendElement(interp, orbListObj, Tcl_NewStringObj(buf, -1));
            }
            Tcl_ListObjAppendElement(interp, waveListObj, orbListObj);
          }
          Tcl_ListObjAppendElement(interp, tcl_result, waveListObj);
        }
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);
       
    } else if (!strcmp(arg, "wavef_tree")) {
      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
      Timestep *ts = mol->get_frame(context.molinfo_frame_number);
      if (ts && ts->qm_timestep && ts->qm_timestep->get_num_wavef() &&
          mol->qm_data && mol->qm_data->get_basis()) {
        char buf[20];                       
        int iwave, orb, i, j, k;
        for (iwave=0; iwave<ts->qm_timestep->get_num_wavef(); iwave++) {
          Tcl_Obj *waveListObj = Tcl_NewListObj(0, NULL);
          int norbitals = ts->qm_timestep->get_num_orbitals(iwave);
          for (orb=0; orb<norbitals; orb++) {
            Tcl_Obj *orbListObj = Tcl_NewListObj(0, NULL);
            for (i=0; i<mol->nAtoms; i++) {
              Tcl_Obj *atomListObj = Tcl_NewListObj(0, NULL);
              for (j=0; j<mol->qm_data->get_basis(i)->numshells; j++) {
                const shell_t *shell = mol->qm_data->get_basis(i, j);
                Tcl_Obj *shellListObj = Tcl_NewListObj(0, NULL);
                Tcl_Obj *angListObj  = Tcl_NewListObj(0, NULL);
                Tcl_Obj *waveListObj = Tcl_NewListObj(0, NULL);
                const float *wave_f = ts->qm_timestep->get_wavecoeffs(iwave) + 
                                      orb*norbitals +
                                      mol->qm_data->get_wave_offset(i, j);
                for (k=0; k<shell->num_cart_func; k++) {
                  char *s = mol->qm_data->get_angular_momentum_str(i, j, k);
                  Tcl_ListObjAppendElement(interp, angListObj, Tcl_NewStringObj(s, -1));
                  delete [] s;
                  sprintf(buf, "%.12f", wave_f[k]);
                  Tcl_ListObjAppendElement(interp, waveListObj, Tcl_NewStringObj(buf, -1));
                }
                sprintf(buf, "%s", mol->qm_data->get_shell_type_str(shell));
                Tcl_ListObjAppendElement(interp, shellListObj, Tcl_NewStringObj(buf, -1));
                Tcl_ListObjAppendElement(interp, shellListObj, angListObj);
                Tcl_ListObjAppendElement(interp, shellListObj, waveListObj);
                Tcl_ListObjAppendElement(interp, atomListObj, shellListObj);
              }
              Tcl_ListObjAppendElement(interp, orbListObj, atomListObj);
            }
            Tcl_ListObjAppendElement(interp, waveListObj, orbListObj);
          }
          Tcl_ListObjAppendElement(interp, tcl_result, waveListObj);
        }
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);
        
    } else if (!strcmp(arg, "qmcharges")) {
      Timestep *ts = mol->get_frame(context.molinfo_frame_number);
      if (ts && ts->qm_timestep && ts->qm_timestep->get_num_charge_sets()) {
        Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
        int i, j;
        for (i=0; i<ts->qm_timestep->get_num_charge_sets(); i++) {
          Tcl_Obj *chargesetObj = Tcl_NewListObj(0, NULL);
          Tcl_Obj *atomListObj = Tcl_NewListObj(0, NULL);
          const double *chargeset = ts->qm_timestep->get_charge_set(i);
          for (j=0; j<mol->nAtoms; j++) {
            Tcl_ListObjAppendElement(interp, atomListObj,
               Tcl_NewDoubleObj(chargeset[j]));
          }
          Tcl_ListObjAppendElement(interp, chargesetObj,
            Tcl_NewStringObj(ts->qm_timestep->get_charge_type_str(i), -1));
          Tcl_ListObjAppendElement(interp, chargesetObj, atomListObj);
          Tcl_ListObjAppendElement(interp, tcl_result, chargesetObj);
        }
        Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), tcl_result);
      }
    generic_molinfo_qm_mat("%.6f", "carthessian", carthessian, 3L*mol->nAtoms);
    generic_molinfo_qm_mat("%.6f", "inthessian",  inthessian,  mol->qm_data->get_num_intcoords());
    generic_molinfo_qm_mat("%.8f", "normalmodes", normalmodes, 3L*mol->nAtoms);
    generic_molinfo_qm_arr("%.2f", "wavenumbers", wavenumbers, 3L*mol->nAtoms);
    generic_molinfo_qm_arr("%.6f", "intensities", intensities, 3L*mol->nAtoms);
    generic_molinfo_qm_arr("%d",   "imagmodes",   imagmodes,   mol->qm_data->get_num_imag());

    } else if (!strcmp(arg, "angles")) {
      Tcl_Obj *alist = Tcl_NewListObj(0, NULL);
      for (int i=0; i<mol->num_angles(); i++) {
        Tcl_Obj *adata = Tcl_NewListObj(0,NULL);
        int type = -1;
        const char *atname;
                    
        if (mol->angleTypes.num() > 0) {
           type = mol->get_angletype(i);
        }

        if (type < 0)
          atname = "unknown";
        else
          atname = mol->angleTypeNames.name(type);
        
        Tcl_ListObjAppendElement(interp, adata, 
                                 Tcl_NewStringObj((char *)atname, -1));
        Tcl_ListObjAppendElement(interp, adata,
                                 Tcl_NewIntObj(mol->angles[3L*i]));
        Tcl_ListObjAppendElement(interp, adata,
                                 Tcl_NewIntObj(mol->angles[3L*i+1]));
        Tcl_ListObjAppendElement(interp, adata,
                                 Tcl_NewIntObj(mol->angles[3L*i+2]));
        Tcl_ListObjAppendElement(interp, alist, adata);
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), alist);
    } else if (!strcmp(arg, "dihedrals")) {
      Tcl_Obj *alist = Tcl_NewListObj(0, NULL);
      for (int i=0; i<mol->num_dihedrals(); i++) {
        Tcl_Obj *adata = Tcl_NewListObj(0,NULL);
        int type = -1;
        const char *atname;
                    
        if (mol->dihedralTypes.num() > 0) {
          type = mol->get_dihedraltype(i);
        }

        if (type < 0)
          atname = "unknown";
        else
          atname = mol->dihedralTypeNames.name(type);
          
        Tcl_ListObjAppendElement(interp, adata, 
                                 Tcl_NewStringObj((char *)atname, -1));
        Tcl_ListObjAppendElement(interp, adata,
                                 Tcl_NewIntObj(mol->dihedrals[4L*i]));
        Tcl_ListObjAppendElement(interp, adata,
                                 Tcl_NewIntObj(mol->dihedrals[4L*i+1]));
        Tcl_ListObjAppendElement(interp, adata,
                                 Tcl_NewIntObj(mol->dihedrals[4L*i+2]));
        Tcl_ListObjAppendElement(interp, adata,
                                 Tcl_NewIntObj(mol->dihedrals[4L*i+3]));
        Tcl_ListObjAppendElement(interp, alist, adata);
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), alist);
    } else if (!strcmp(arg, "impropers")) {
      Tcl_Obj *alist = Tcl_NewListObj(0, NULL);
      for (int i=0; i<mol->num_impropers(); i++) {
        Tcl_Obj *adata = Tcl_NewListObj(0,NULL);
        int type = -1;
        const char *atname;
                    
        if (mol->improperTypes.num() > 0) {
          type = mol->get_impropertype(i);
        }

        if (type < 0)
          atname = "unknown";
        else
          atname = mol->improperTypeNames.name(type);
          
        Tcl_ListObjAppendElement(interp, adata, 
                                 Tcl_NewStringObj((char *)atname, -1));
        Tcl_ListObjAppendElement(interp, adata,
                                 Tcl_NewIntObj(mol->impropers[4L*i]));
        Tcl_ListObjAppendElement(interp, adata,
                                 Tcl_NewIntObj(mol->impropers[4L*i+1]));
        Tcl_ListObjAppendElement(interp, adata,
                                 Tcl_NewIntObj(mol->impropers[4L*i+2]));
        Tcl_ListObjAppendElement(interp, adata,
                                 Tcl_NewIntObj(mol->impropers[4L*i+3]));
        Tcl_ListObjAppendElement(interp, alist, adata);
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), alist);
    } else if (!strcmp(arg, "crossterms")) {
      Tcl_Obj *alist = Tcl_NewListObj(0, NULL);
      for (int i=0; i<mol->num_cterms(); i++) {
        Tcl_Obj *adata = Tcl_NewListObj(0,NULL);
        for (int j=0; j<8; j++) {
          Tcl_ListObjAppendElement(interp, adata, Tcl_NewIntObj(mol->cterms[8L*i+j]));
        }
        Tcl_ListObjAppendElement(interp, alist, adata);
      }
      Tcl_ListObjAppendElement(interp, Tcl_GetObjResult(interp), alist);
    } else {
      Tcl_ResetResult(interp);
      Tcl_AppendResult(interp, "molinfo: cannot find molinfo attribute '",
                       argv[term], "'", NULL);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}


#define generic_molinfo_set_data(name, func1, func2) \
} else if (!strcmp(argv[term], name)) { \
int onoff; \
if (Tcl_GetBoolean(interp, data[term], &onoff) != TCL_OK) return TCL_ERROR; \
if (onoff) { func1 ; } else { func2 ; }

#define generic_molinfo_simulation_set(name, type)  \
} else if (!strcmp(argv[term], name)) { \
Timestep *ts = mol->get_frame(context.molinfo_frame_number); \
if (ts) ts->energy[type] = (float) atof(data[term]);

#define generic_molinfo_pbc_set(name, type) \
} else if (!strcmp(argv[term], name)) { \
Timestep *ts = mol->get_frame(context.molinfo_frame_number); \
if (ts) { ts->type = (float) atof(data[term]); mol->change_pbc(); }

// given the list of strings and data, set the right values
static int molinfo_set(ctxt context, int molid, int argc, const char *argv[],
                       const char *data[], Tcl_Interp *interp, int frame_num) {
  // does the molecule exist?
  context.molinfo_get_id = molid;
  context.molinfo_get_index = context.mlist->mol_index_from_id(molid);

  if (context.molinfo_get_index == -1) {
    char s[10];
    sprintf(s, "%d", molid);
    Tcl_AppendResult(interp, "molinfo: set: no molecule exists with id ",
                     s, NULL);
    return TCL_ERROR;
  }

  // get the right frame number
  switch (frame_num) {
    case AtomSel::TS_NOW: 
      context.molinfo_frame_number =
        context.mlist->molecule(context.molinfo_get_index) -> frame();
      break;
    case AtomSel::TS_LAST:
      context.molinfo_frame_number = 
        context.mlist->molecule(context.molinfo_get_index) -> numframes()-1;
      break;
    default:      
      context.molinfo_frame_number = frame_num;
  }
  Molecule *mol = context.mlist->molecule(context.molinfo_get_index);
  
  for (int term=0; term<argc; term++) {
    context.molinfo_get_string = argv[term];
    if (!strcmp(argv[term], "center")) {
      float x, y, z;
      if (sscanf((const char *)data[term], "%f %f %f", &x, &y, &z) != 3) {
        Tcl_AppendResult(interp, 
            "molinfo: set center: must have three position elements", NULL);
        return TCL_ERROR;
      }
      mol->change_center(x, y, z);
    } else if (!strcmp(argv[term], "center_matrix")) {
      float mat[16];
      if (!read_matrix(interp, data[term], mat)) return TCL_ERROR;
      mol->set_cent_trans(mat[12], mat[13], mat[14]);
    } else if (!strcmp(argv[term], "rotate_matrix")) {
      Matrix4 mat;
      if (!read_matrix(interp, data[term], mat.mat)) return TCL_ERROR;
      mol->set_rot(mat);
    } else if (!strcmp(argv[term], "scale_matrix")) {
      float mat[16];
      if (!read_matrix(interp, data[term], mat)) return TCL_ERROR;
      mol->set_scale(mat[0]);
    } else if (!strcmp(argv[term], "global_matrix")) {
      float mat[16];
      if (!read_matrix(interp, data[term], mat)) return TCL_ERROR;
      mol->set_glob_trans(mat[12], mat[13], mat[14]);
    } else if (!strcmp(argv[term], "angles")) {
      mol->clear_angles();
      mol->angleTypeNames.clear();
      mol->set_dataset_flag(BaseMolecule::ANGLES);

      Tcl_Obj *alist = Tcl_NewListObj(0,NULL);
      Tcl_SetStringObj(alist, data[term], -1);
      Tcl_IncrRefCount(alist); 

      int numangles;
      Tcl_Obj **adata;
      Tcl_ListObjGetElements(interp, alist, &numangles, &adata);

      for (int i=0; i < numangles; i++) {
        int numentries;
        Tcl_Obj **a;

        Tcl_ListObjGetElements(interp, adata[i], &numentries, &a);
        if (numentries != 4) {
          Tcl_AppendResult(interp, "molinfo: incorrect data item for "
                           "'set angles' :", Tcl_GetString(adata[i]), NULL);
          return TCL_ERROR;
        }
        int type = mol->angleTypeNames.add_name(Tcl_GetString(a[0]), 0);
        int a1, a2, a3;
        if (Tcl_GetIntFromObj(interp, a[1], &a1) != TCL_OK) return TCL_ERROR;
        if (Tcl_GetIntFromObj(interp, a[2], &a2) != TCL_OK) return TCL_ERROR;
        if (Tcl_GetIntFromObj(interp, a[3], &a3) != TCL_OK) return TCL_ERROR;
        mol->add_angle(a1, a2, a3, type);
      }
      // release storage
      Tcl_InvalidateStringRep(alist);
      Tcl_DecrRefCount(alist); 

    } else if (!strcmp(argv[term], "dihedrals")) { 
      mol->clear_dihedrals();
      mol->dihedralTypeNames.clear();
      mol->set_dataset_flag(BaseMolecule::ANGLES);

      Tcl_Obj *alist = Tcl_NewListObj(0,NULL);
      Tcl_SetStringObj(alist, data[term], -1);
      Tcl_IncrRefCount(alist); 

      int numdihedrals;
      Tcl_Obj **adata;
      Tcl_ListObjGetElements(interp, alist, &numdihedrals, &adata);
      for (int i=0; i < numdihedrals; i++) {
        int numentries;
        Tcl_Obj **a;
        Tcl_ListObjGetElements(interp, adata[i], &numentries, &a);
        if (numentries != 5) {
          Tcl_AppendResult(interp, "molinfo: incorrect data item for "
                           "'set dihedrals' :", Tcl_GetString(adata[i]), NULL);
          return TCL_ERROR;
        }
        int type = mol->dihedralTypeNames.add_name(Tcl_GetString(a[0]), 0);
        int a1, a2, a3, a4;
        if (Tcl_GetIntFromObj(interp, a[1], &a1) != TCL_OK) return TCL_ERROR;
        if (Tcl_GetIntFromObj(interp, a[2], &a2) != TCL_OK) return TCL_ERROR;
        if (Tcl_GetIntFromObj(interp, a[3], &a3) != TCL_OK) return TCL_ERROR;
        if (Tcl_GetIntFromObj(interp, a[4], &a4) != TCL_OK) return TCL_ERROR;
        mol->add_dihedral(a1, a2, a3, a4, type);
      }
      // release storage
      Tcl_InvalidateStringRep(alist);
      Tcl_DecrRefCount(alist); 

    } else if (!strcmp(argv[term], "impropers")) {
      mol->clear_impropers();
      mol->improperTypeNames.clear();
      mol->set_dataset_flag(BaseMolecule::ANGLES);

      Tcl_Obj *alist = Tcl_NewListObj(0,NULL);
      Tcl_SetStringObj(alist, data[term], -1);
      Tcl_IncrRefCount(alist); 

      int numimpropers;
      Tcl_Obj **adata;
      Tcl_ListObjGetElements(interp, alist, &numimpropers, &adata);
      for (int i=0; i < numimpropers; i++) {
        int numentries;
        Tcl_Obj **a;
        Tcl_ListObjGetElements(interp, adata[i], &numentries, &a);
        if (numentries != 5) {
          Tcl_AppendResult(interp, "molinfo: incorrect data item for "
                           "'set impropers' :", Tcl_GetString(adata[i]), NULL);
          return TCL_ERROR;
        }
        int type = mol->improperTypeNames.add_name(Tcl_GetString(a[0]), 0);
        int a1, a2, a3, a4;
        if (Tcl_GetIntFromObj(interp, a[1], &a1) != TCL_OK) return TCL_ERROR;
        if (Tcl_GetIntFromObj(interp, a[2], &a2) != TCL_OK) return TCL_ERROR;
        if (Tcl_GetIntFromObj(interp, a[3], &a3) != TCL_OK) return TCL_ERROR;
        if (Tcl_GetIntFromObj(interp, a[4], &a4) != TCL_OK) return TCL_ERROR;
        mol->add_improper(a1, a2, a3, a4, type);
      }
      // release storage
      Tcl_InvalidateStringRep(alist);
      Tcl_DecrRefCount(alist); 
    } else if (!strcmp(argv[term], "crossterms")) {
      mol->clear_cterms();
      mol->set_dataset_flag(BaseMolecule::CTERMS);

      Tcl_Obj *alist = Tcl_NewListObj(0,NULL);
      Tcl_SetStringObj(alist, data[term], -1);
      Tcl_IncrRefCount(alist); 

      int numcterms;
      Tcl_Obj **adata;
      Tcl_ListObjGetElements(interp, alist, &numcterms, &adata);
      for (int i=0; i < numcterms; i++) {
        int numentries;
        Tcl_Obj **a;
        Tcl_ListObjGetElements(interp, adata[i], &numentries, &a);
        if (numentries != 8) {
          Tcl_AppendResult(interp, "molinfo: incorrect data item for "
                           "'set crossterms' :", Tcl_GetString(adata[i]), NULL);
          return TCL_ERROR;
        }
        int a1, a2, a3, a4, a5, a6, a7, a8;
        if (Tcl_GetIntFromObj(interp, a[0], &a1) != TCL_OK) return TCL_ERROR;
        if (Tcl_GetIntFromObj(interp, a[1], &a2) != TCL_OK) return TCL_ERROR;
        if (Tcl_GetIntFromObj(interp, a[2], &a3) != TCL_OK) return TCL_ERROR;
        if (Tcl_GetIntFromObj(interp, a[3], &a4) != TCL_OK) return TCL_ERROR;
        if (Tcl_GetIntFromObj(interp, a[4], &a5) != TCL_OK) return TCL_ERROR;
        if (Tcl_GetIntFromObj(interp, a[5], &a6) != TCL_OK) return TCL_ERROR;
        if (Tcl_GetIntFromObj(interp, a[6], &a7) != TCL_OK) return TCL_ERROR;
        if (Tcl_GetIntFromObj(interp, a[7], &a8) != TCL_OK) return TCL_ERROR;
        mol->add_cterm(a1, a2, a3, a4, a5, a6, a7, a8);
      }
      // release storage
      Tcl_InvalidateStringRep(alist);
      Tcl_DecrRefCount(alist); 
    } else if (!strcmp(argv[term], "frame")) {
      // XXX this isn't correctly sending events to the GUI,
      // so if you do 'molinfo top set frame 5' the GUI is 
      // out of date
      //
      // This is correct and intended behavior; if you want GUI events,
      // the command to use is "animate goto <frame>. -- JRG 12/27/07
      mol->override_current_frame(atoi(data[term]));
      mol->change_ts();
      
    generic_molinfo_set_data("active", context.app->molecule_activate(mol->id(), 1), context.app->molecule_activate(mol->id(), 0))
    generic_molinfo_set_data("drawn", context.app->molecule_display(mol->id(), 1), context.app->molecule_display(mol->id(), 0))
    generic_molinfo_set_data("displayed", context.app->molecule_display(mol->id(), 1), context.app->molecule_display(mol->id(), 0))
    generic_molinfo_set_data("fixed", context.app->molecule_fix(mol->id(), 1), context.app->molecule_fix(mol->id(), 0))
    generic_molinfo_set_data("top", context.app->molecule_make_top(mol->id()), Tcl_SetResult(interp, (char *) "Cannot set 'top' to false.", TCL_STATIC); return TCL_ERROR)

    generic_molinfo_simulation_set("bond", TSE_BOND)
    generic_molinfo_simulation_set("angle", TSE_ANGLE)
    generic_molinfo_simulation_set("dihedral", TSE_DIHE)
    generic_molinfo_simulation_set("improper", TSE_IMPR)
    generic_molinfo_simulation_set("vdw", TSE_VDW)
    generic_molinfo_simulation_set("electrostatic", TSE_COUL)
    generic_molinfo_simulation_set("elec", TSE_COUL)
    generic_molinfo_simulation_set("hbond", TSE_HBOND)
    generic_molinfo_simulation_set("kinetic", TSE_KE)
    generic_molinfo_simulation_set("potential", TSE_PE)
    generic_molinfo_simulation_set("temperature", TSE_TEMP)
    generic_molinfo_simulation_set("temp", TSE_TEMP)
    generic_molinfo_simulation_set("energy", TSE_TOTAL)
    generic_molinfo_simulation_set("volume", TSE_VOLUME)
    generic_molinfo_simulation_set("pressure", TSE_PRESSURE)
    generic_molinfo_simulation_set("efield", TSE_EFIELD)
    generic_molinfo_simulation_set("urey_bradley", TSE_UREY_BRADLEY)
    generic_molinfo_simulation_set("molinfo_restraint", TSE_RESTRAINT)

    } else if (!strcmp(argv[term], "timesteps")) {
      Timestep *ts = mol->get_frame(context.molinfo_frame_number);
      if (ts) ts->timesteps = atoi(data[term]);

    generic_molinfo_pbc_set("a", a_length)
    generic_molinfo_pbc_set("b", b_length)
    generic_molinfo_pbc_set("c", c_length)
    generic_molinfo_pbc_set("alpha", alpha)
    generic_molinfo_pbc_set("beta", beta)
    generic_molinfo_pbc_set("gamma", gamma)
    generic_molinfo_pbc_set("physical_time", physical_time)

    } else {
      Tcl_AppendResult(interp, "molinfo: cannot find molinfo attribute '",
                       argv[term], "'", NULL);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}


// Function:  molinfo
// Option  :  molinfo num
//  Returns:   number of molecules
// Option  :  molinfo index <int>
//  Returns:   molecule id of the nth molecule (starting at index = 0)
// Option  :  molinfo list
//  Returns:   list of all molecule ids
// Option  :  molinfo top
//  Returns:   molecule id of the 'top' molecule
// Option  :  molinfo {molecule number} get <data>
//  Returns:   the given data for that molecule
// Option  :  molinfo {molecule number} set <data> <data fields>
//  Does (okay, this isn't a 'info' thing): sets the data field(s)

int molecule_tcl(ClientData data, Tcl_Interp *interp, int argc, const char *argv[])
{
  VMDApp *app = (VMDApp *)data;
  // set context variable here
  ctxt context;
  context.mlist = app->moleculeList;
  context.matlist = app->materialList;
  context.app = app;

  if (argc == 1) {
    Tcl_SetResult(interp,
      (char *)
      "usage: molinfo <command> [args...]\n\n"
      "Commands:"
      "\nMolecule IDs:\n"
      "  list                  -- lists all existing molecule IDs\n"
      "  num                   -- number of loaded molecules\n"
      "  top                   -- gets ID of top molecule (or -1 if none)\n"
      "  index <n>             -- gets ID of n-th molecule\n"
      "\nGetting and Setting Molecular Information:\n" 
     // "  keywords              -- returns a list of molinfo keywords\n"     // XXX obsolete???
      "  <molid> get <(list of) keywords>\n"
      "  <molid> set <(list of) keywords> <(list of) values>\n",
      TCL_STATIC);
    return TCL_ERROR;
  }

  // what does it want?
  if (argc == 2) {

// Option  :  molinfo num
//  Returns:   number of molecules
    if (!strcmp(argv[1], "num")) {
      char tmpstring[64];
      sprintf(tmpstring, "%d", context.mlist->num());
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      return TCL_OK;
    }

// Option  :  molinfo list
//  Returns:   list of all molecule ids
    if (!strcmp(argv[1], "list")) {
      if (context.mlist->num() <= 0) {
        return TCL_OK;
      }
      char s[20];
      sprintf(s, "%d", context.mlist->molecule(0)->id());
      Tcl_AppendResult(interp, s, (char *) NULL);
      for (int i=1; i<context.mlist -> num(); i++) {
        sprintf(s, "%d", context.mlist->molecule(i)->id());
        Tcl_AppendResult(interp, " ", s, (char *) NULL);
      }
      return TCL_OK;
    }
// Option  :  molinfo top
//  Returns:   molecule id of the 'top' molecule
    if (!strcmp(argv[1], "top")) {
      if (context.mlist->top()) {
        char tmpstring[64];
        sprintf(tmpstring, "%d", context.mlist->top()->id());
        Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      } else {
        Tcl_SetResult(interp, (char *) "-1", TCL_STATIC);
      }
      return TCL_OK;
    }

    // otherwise, I don't know
    Tcl_AppendResult(interp, "molinfo: couldn't understand '",
                     argv[1], "'", NULL);
    return TCL_ERROR;
  } // end of commands with only one option

  if (argc == 3) { // commands with two options
    int val;
    if (Tcl_GetInt(interp, argv[2], &val) != TCL_OK) {
      return TCL_ERROR;
    }
// Option  :  molecule index <int>
//  Returns:   molecule id of the nth molecule (starting at index = 0)
    if (!strcmp(argv[1], "index")) {
      if (context.mlist->molecule(val)) {
        char tmpstring[64];
        sprintf(tmpstring, "%d", context.mlist->molecule(val)->id());
        Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      } else {
        Tcl_SetResult(interp, (char *) "-1", TCL_STATIC);
      }
      return TCL_OK;
    }
    Tcl_AppendResult(interp, "molinfo: couldn't understand '",
                     argv[1], "'", NULL);
    return TCL_ERROR;
  }
// Option  :  molinfo {molecule number} get <data> [frame <number>]
//  Returns:   the given data for that molecule
  if ((argc == 4 && !strcmp(argv[2], "get")) ||
      (argc == 6 && !strcmp(argv[2], "get") && !strcmp(argv[4], "frame"))) {
    int frame_num;
    if (argc == 4) {
      frame_num = AtomSel::TS_NOW;
    } else {
      if (AtomSel::get_frame_value(argv[5], &frame_num) != 0) {
        Tcl_SetResult(interp, (char *)
          "atomselect: bad frame number in input, must be "
          "'first', 'last', 'now', or a non-negative number", TCL_STATIC);
        return TCL_ERROR;
      }
    }
    int val;
    // get the molecule name recursively
    if (!strcmp(argv[1], "top")) {
      if (Tcl_VarEval(interp, argv[0], " top", NULL) != TCL_OK ||
          Tcl_GetInt(interp, Tcl_GetStringResult(interp), &val) != TCL_OK     ) {
        return TCL_ERROR;
      }
    } else {
      if (Tcl_GetInt(interp, argv[1], &val) != TCL_OK) {
        return TCL_ERROR;
      }
    }
    Tcl_ResetResult(interp);

    // split the data into the various terms
    const char **list;
    int num_list;
    if (Tcl_SplitList(interp, argv[3], &num_list, &list) != TCL_OK) {
      return TCL_ERROR;
    }
    // and return the information
    int result = molinfo_get(context, val, num_list, list, interp, frame_num);
    ckfree((char *) list); // free of tcl data

    return result;
  }
// Option  :  molinfo {molecule number} set <data> <new data> [frame <number>]
//  Does   :   sets the given data for that molecule
  if ((argc == 5 && !strcmp(argv[2], "set")) ||
      (argc == 7 && !strcmp(argv[2], "set") && !strcmp(argv[5], "frame"))) {
    // get the frame number
    int frame_num;
    if (argc == 5) {
      frame_num = AtomSel::TS_NOW;
    } else {
      if (AtomSel::get_frame_value(argv[6], &frame_num) != 0) {
        Tcl_SetResult(interp, (char *)
          "atomselect: bad frame number in input, must be "
          "'first', 'last', 'now', or a non-negative number", TCL_STATIC);
        return TCL_ERROR;
      }
    }

    int val;
    if (!strcmp(argv[1], "top")) {
      if (Tcl_VarEval(interp, argv[0], " top", NULL) != TCL_OK ||
          Tcl_GetInt(interp, Tcl_GetStringResult(interp), &val) != TCL_OK     ) {
        return TCL_ERROR;
      }
    } else {
      if (Tcl_GetInt(interp, argv[1], &val) != TCL_OK) {
        return TCL_ERROR;
      }
    }

    Tcl_ResetResult(interp);

    // make sure the two lists have the same number of terms
    const char **list1, **list2;
    int num_list1, num_list2;
    if (Tcl_SplitList(interp, argv[3], &num_list1, &list1) != TCL_OK) {
      return TCL_ERROR;
    }
    if (Tcl_SplitList(interp, argv[4], &num_list2, &list2) != TCL_OK) {
      ckfree((char *)list1); // free of tcl data
      return TCL_ERROR;
    }
    if (num_list1 != num_list2) {
      ckfree((char *)list1); // free of tcl data
      ckfree((char *)list2); // free of tcl data
      Tcl_SetResult(interp, (char *) "molinfo: set: argument and value lists have different sizes", TCL_STATIC);
      return TCL_ERROR;
    }

    // call the 'set' routine
    int result = molinfo_set(context, val, num_list1, list1, list2, interp, frame_num);

    ckfree((char *)list1); // free of tcl data
    ckfree((char *)list2); // free of tcl data
    return result;
  }

  // There's been an error; find out what's wrong
  Tcl_SetResult(interp, (char *) "molinfo: called with unknown command", TCL_STATIC);
  if (argc >= 3) {
    if (!strcmp(argv[2], "get")) {
      Tcl_SetResult(interp, (char *) "molinfo: incorrect format for 'get'", 
                    TCL_STATIC);
    } else if (!strcmp(argv[2], "set")) {
      Tcl_SetResult(interp, (char *) "molinfo: incorrect format for 'set'",
                    TCL_STATIC);
    }
  } else if (argc >= 2) {
    if (!strcmp(argv[1], "get") || strcmp(argv[1], "set")) {
      Tcl_SetResult(interp, (char *) "molinfo: missing molecule number",
                    TCL_STATIC);
    }
  }

  return TCL_ERROR;
}
