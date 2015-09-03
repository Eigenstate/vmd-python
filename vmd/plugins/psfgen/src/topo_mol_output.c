
#include <string.h>
#include <stdlib.h>
#include "topo_mol_output.h"
#include "topo_mol_struct.h"
#include "pdb_file.h"

int topo_mol_write_pdb(topo_mol *mol, FILE *file, void *v, 
                                void (*print_msg)(void *, const char *)) {

  char buf[128], insertion[2];
  int iseg,nseg,ires,nres,atomid,resid;
  int has_guessed_atoms = 0;
  double x,y,z,o,b;
  topo_mol_segment_t *seg;
  topo_mol_residue_t *res;
  topo_mol_atom_t *atom;

  if ( ! mol ) return -1;

  write_pdb_remark(file,"original generated coordinate pdb file");

  atomid = 0;
  nseg = hasharray_count(mol->segment_hash);
  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = mol->segment_array[iseg];
    if (! seg) continue;

    if ( strlen(seg->segid) > 4 ) {
      sprintf(buf,
	"warning: truncating segid %s to 4 characters allowed by pdb format",
	seg->segid);
      print_msg(v,buf);
    }

    nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      res = &(seg->residue_array[ires]);
      for ( atom = res->atoms; atom; atom = atom->next ) {
        /* Paranoid: make sure x,y,z,o are set. */
        x = y = z = 0.0; o = -1.0;
        ++atomid;
        switch ( atom->xyz_state ) {
        case TOPO_MOL_XYZ_SET:
          x = atom->x;  y = atom->y;  z = atom->z;  o = 1.0;
          break;
        case TOPO_MOL_XYZ_GUESS:
        case TOPO_MOL_XYZ_BADGUESS:
          x = atom->x;  y = atom->y;  z = atom->z;  o = 0.0;
          has_guessed_atoms = 1;
          break;
        default:
          print_msg(v,"ERROR: Internal error, atom has invalid state.");
          print_msg(v,"ERROR: Treating as void.");
          /* Yes, fall through */
        case TOPO_MOL_XYZ_VOID:
          x = y = z = 0.0;  o = -1.0;
          break;
        }
        b = atom->partition;
        insertion[0] = 0;
        insertion[1] = 0;
        sscanf(res->resid, "%d%c", &resid, insertion);
        write_pdb_atom(file,atomid,atom->name,res->name,resid,insertion,
		(float)x,(float)y,(float)z,(float)o,(float)b,res->chain,
		seg->segid,atom->element);
      }
    }
  }

  write_pdb_end(file);
  if (has_guessed_atoms) {
    print_msg(v, 
        "Info: Atoms with guessed coordinates will have occupancy of 0.0.");
  }
  return 0;
}

int topo_mol_write_namdbin(topo_mol *mol, FILE *file, void *v, 
                                void (*print_msg)(void *, const char *)) {

  char static_assert_int_is_32_bits[sizeof(int) == 4 ? 1 : -1];
  int iseg,nseg,ires,nres,atomid,resid;
  int has_void_atoms = 0;
  int numatoms;
  double x,y,z,xyz[3];
  topo_mol_segment_t *seg;
  topo_mol_residue_t *res;
  topo_mol_atom_t *atom;

  if ( ! mol ) return -1;

  numatoms = 0;
  nseg = hasharray_count(mol->segment_hash);
  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = mol->segment_array[iseg];
    if (! seg) continue;
    nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      res = &(seg->residue_array[ires]);
      for ( atom = res->atoms; atom; atom = atom->next ) {
        ++numatoms;
      }
    }
  }
  if ( fwrite(&numatoms, sizeof(int), 1, file) != 1 ) {
    print_msg(v, "error writing namdbin file");
    return -2;
  }
  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = mol->segment_array[iseg];
    if (! seg) continue;
    nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      res = &(seg->residue_array[ires]);
      for ( atom = res->atoms; atom; atom = atom->next ) {
        /* Paranoid: make sure x,y,z are set. */
        x = y = z = 0.0;
        switch ( atom->xyz_state ) {
        case TOPO_MOL_XYZ_SET:
        case TOPO_MOL_XYZ_GUESS:
        case TOPO_MOL_XYZ_BADGUESS:
          x = atom->x;  y = atom->y;  z = atom->z;
          break;
        default:
          print_msg(v,"ERROR: Internal error, atom has invalid state.");
          print_msg(v,"ERROR: Treating as void.");
          /* Yes, fall through */
        case TOPO_MOL_XYZ_VOID:
          x = y = z = 0.0;
          has_void_atoms = 1;
          break;
        }
        xyz[0] = x;
        xyz[1] = y;
        xyz[2] = z;
        if ( fwrite(xyz, sizeof(double), 3, file) != 3 ) {
          print_msg(v, "error writing namdbin file");
          return -3;
        }
      }
    }
  }

  if (has_void_atoms) {
    print_msg(v, 
        "Warning: Atoms with unknown coordinates written at 0. 0. 0.");
  }
  return 0;
}

int topo_mol_write_psf(topo_mol *mol, FILE *file, int charmmfmt, int nocmap,
                      void *v, void (*print_msg)(void *, const char *)) {

  char buf[128];
  int iseg,nseg,ires,nres,atomid;
  int namdfmt, charmmext;
  topo_mol_segment_t *seg;
  topo_mol_residue_t *res;
  topo_mol_atom_t *atom;
  topo_mol_bond_t *bond;
  int nbonds;
  topo_mol_angle_t *angl;
  int nangls;
  topo_mol_dihedral_t *dihe;
  int ndihes;
  topo_mol_improper_t *impr;
  int nimprs;
  topo_mol_cmap_t *cmap;
  int ncmaps;
  int numinline;
  int npres,ipres,ntopo,itopo;
  topo_defs_topofile_t *topo;
  topo_mol_patch_t *patch;
  topo_mol_patchres_t *patchres;
  char defpatch[10];
  fpos_t ntitle_pos, save_pos;
  char ntitle_fmt[128];
  int ntitle_count;
  strcpy(defpatch,"");

  if ( ! mol ) return -1;

  namdfmt = 0;
  charmmext = 0;
  atomid = 0;
  nbonds = 0;
  nangls = 0;
  ndihes = 0;
  nimprs = 0;
  ncmaps = 0;
  nseg = hasharray_count(mol->segment_hash);
  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = mol->segment_array[iseg];
    if (! seg) continue;

    if ( strlen(seg->segid) > 4 ) {
      charmmext = 1;
    }

    nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      res = &(seg->residue_array[ires]);
      if (strlen(res->resid) > 4) {
        charmmext = 1;
      }
      for ( atom = res->atoms; atom; atom = atom->next ) {
        atom->atomid = ++atomid;
        if (strlen(atom->name) > 4) {
          charmmext = 1;
        }
        if ((! charmmfmt) && (strlen(atom->type) > 4)) {
          charmmext = 1;
        }
        if ((! charmmfmt) && (strlen(atom->type) > 6)) {
          namdfmt = 1;
        }
        for ( bond = atom->bonds; bond;
                bond = topo_mol_bond_next(bond,atom) ) {
          if ( bond->atom[0] == atom && ! bond->del ) {
            ++nbonds;
          }
        }
        for ( angl = atom->angles; angl;
                angl = topo_mol_angle_next(angl,atom) ) {
          if ( angl->atom[0] == atom && ! angl->del ) {
            ++nangls;
          }
        }
        for ( dihe = atom->dihedrals; dihe;
                dihe = topo_mol_dihedral_next(dihe,atom) ) {
          if ( dihe->atom[0] == atom && ! dihe->del ) {
            ++ndihes;
          }
        }
        for ( impr = atom->impropers; impr;
                impr = topo_mol_improper_next(impr,atom) ) {
          if ( impr->atom[0] == atom && ! impr->del ) {
            ++nimprs;
          }
        }
        for ( cmap = atom->cmaps; cmap;
                cmap = topo_mol_cmap_next(cmap,atom) ) {
          if ( cmap->atom[0] == atom && ! cmap->del ) {
            ++ncmaps;
          }
        }
      }
    }
  }
  sprintf(buf,"total of %d atoms",atomid);
  print_msg(v,buf);
  sprintf(buf,"total of %d bonds",nbonds);
  print_msg(v,buf);
  sprintf(buf,"total of %d angles",nangls);
  print_msg(v,buf);
  sprintf(buf,"total of %d dihedrals",ndihes);
  print_msg(v,buf);
  sprintf(buf,"total of %d impropers",nimprs);
  print_msg(v,buf);

  if ( namdfmt ) { charmmext = 0; }
  else if ( atomid > 9999999 ) { charmmext = 1; }

  if ( namdfmt ) {
    print_msg(v,"Structure requires space-delimited NAMD PSF format");
  } else if ( charmmext ) {
    print_msg(v,"Structure requires EXTended PSF format");
  }

  ntitle_fmt[0] = '\0';
  strcat(ntitle_fmt, "PSF");
  if ( namdfmt ) strcat(ntitle_fmt, " NAMD");
  if ( charmmext ) strcat(ntitle_fmt, " EXT");
  if ( nocmap ) {
    sprintf(buf,"total of %d cross-terms (not written to file)",ncmaps);
  } else {
    sprintf(buf,"total of %d cross-terms",ncmaps);
    if ( ncmaps ) {
      strcat(ntitle_fmt, " CMAP");
    } else {
      nocmap = 1;
    }
  }
  print_msg(v,buf);
  strcat(ntitle_fmt, "\n\n%8d !NTITLE\n");

  fgetpos(file,&ntitle_pos);
  fprintf(file,ntitle_fmt,1);
  ntitle_count = 1;
  if ( charmmfmt ) 
   fprintf(file," REMARKS %s\n","original generated structure charmm psf file");
  else
   fprintf(file," REMARKS %s\n","original generated structure x-plor psf file");
  
  if (mol->npatch) {
    ntitle_count++;
    fprintf(file," REMARKS %i patches were applied to the molecule.\n", mol->npatch);
  }

  ntopo = hasharray_count(mol->defs->topo_hash);
  for ( itopo=0; itopo<ntopo; ++itopo ) {
    topo = &(mol->defs->topo_array[itopo]);
    ntitle_count++;
    fprintf(file," REMARKS topology %s \n", topo->filename);
  }

  nseg = hasharray_count(mol->segment_hash);
  for ( iseg=0; iseg<nseg; ++iseg ) {
    char angles[20], diheds[20];
    seg = mol->segment_array[iseg];
    if (! seg) continue;
    strcpy(angles,"none");
    strcpy(diheds,"");
    if (seg->auto_angles)    strcpy(angles,"angles");
    if (seg->auto_dihedrals) strcpy(diheds,"dihedrals");
    ntitle_count++;
    fprintf(file," REMARKS segment %s { first %s; last %s; auto %s %s }\n", seg->segid, seg->pfirst, seg->plast, angles, diheds);
  }

  for ( patch = mol->patches; patch; patch = patch->next ) {
    strcpy(defpatch,"");
    if (patch->deflt) strcpy(defpatch,"default");
    npres = patch->npres;
    ipres = 0;
    for ( patchres = patch->patchresids; patchres; patchres = patchres->next ) {
      /* Test the existence of segid:resid for the patch */
      if (!topo_mol_validate_patchres(mol,patch->pname,patchres->segid, patchres->resid)) {
	break;
      }
    }
    if ( patchres ) continue;

    for ( patchres = patch->patchresids; patchres; patchres = patchres->next ) {
      if (ipres==0) {
        ntitle_count++;
        fprintf(file," REMARKS %spatch %s ", defpatch, patch->pname);
      }
      if (ipres>0 && !ipres%6) {
        ntitle_count++;
        fprintf(file,"\n REMARKS patch ---- ");
      }
      fprintf(file,"%s:%s  ", patchres->segid, patchres->resid);
      if (ipres==npres-1) fprintf(file,"\n");
      ipres++;
    }
  }
  fprintf(file,"\n");
  fgetpos(file,&save_pos);
  fsetpos(file,&ntitle_pos);
  fprintf(file,ntitle_fmt,ntitle_count);
  fsetpos(file,&save_pos);

  fprintf(file,"%8d !NATOM\n",atomid);
  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = mol->segment_array[iseg];
    if (! seg) continue;
    nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      char resid[9];
      res = &(seg->residue_array[ires]);
      strncpy(resid,res->resid,9);
      resid[ charmmext ? 8 : 4 ] = '\0';
      if ( charmmfmt ) for ( atom = res->atoms; atom; atom = atom->next ) {
        int idef,typeid;
        idef = hasharray_index(mol->defs->type_hash,atom->type);
        if ( idef == HASHARRAY_FAIL ) {
          sprintf(buf,"unknown atom type %s",atom->type);
          print_msg(v,buf);
          return -3;
        }
        typeid = mol->defs->type_array[idef].id;
        fprintf(file, ( charmmext ?
                     "%10d %-8s %-8s %-8s %-8s %4d %10.6f    %10.4f  %10d\n" :
                     "%8d %-4s %-4s %-4s %-4s %4d %10.6f    %10.4f  %10d\n" ),
                atom->atomid, seg->segid,resid,res->name,
                atom->name,typeid,atom->charge,atom->mass,0);
      } else for ( atom = res->atoms; atom; atom = atom->next ) {
        fprintf(file, ( charmmext ?
                     "%10d %-8s %-8s %-8s %-8s %-6s %10.6f    %10.4f  %10d\n" :
                     "%8d %-4s %-4s %-4s %-4s %-4s %10.6f    %10.4f  %10d\n" ),
                atom->atomid, seg->segid,resid,res->name,
                atom->name,atom->type,atom->charge,atom->mass,0);
      }
    }
  }
  fprintf(file,"\n");

  fprintf(file,"%8d !NBOND: bonds\n",nbonds);
  numinline = 0;
  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = mol->segment_array[iseg];
    if (! seg) continue;
    nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      res = &(seg->residue_array[ires]);
      for ( atom = res->atoms; atom; atom = atom->next ) {
        for ( bond = atom->bonds; bond;
                bond = topo_mol_bond_next(bond,atom) ) {
          if ( bond->atom[0] == atom && ! bond->del ) {
            if ( numinline == 4 ) { fprintf(file,"\n");  numinline = 0; }
            fprintf(file, ( charmmext ? " %9d %9d" : " %7d %7d"),
                    atom->atomid,bond->atom[1]->atomid);
            ++numinline;
          }
        }
      }
    }
  }
  fprintf(file,"\n\n");

  fprintf(file,"%8d !NTHETA: angles\n",nangls);
  numinline = 0;
  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = mol->segment_array[iseg];
    if (! seg) continue;
    nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      res = &(seg->residue_array[ires]);
      for ( atom = res->atoms; atom; atom = atom->next ) {
        for ( angl = atom->angles; angl;
                angl = topo_mol_angle_next(angl,atom) ) {
          if ( angl->atom[0] == atom && ! angl->del ) {
            if ( numinline == 3 ) { fprintf(file,"\n");  numinline = 0; }
            fprintf(file, ( charmmext ? " %9d %9d %9d" : " %7d %7d %7d"),atom->atomid,
                angl->atom[1]->atomid,angl->atom[2]->atomid);
            ++numinline;
          }
        }
      }
    }
  }
  fprintf(file,"\n\n");

  fprintf(file,"%8d !NPHI: dihedrals\n",ndihes);
  numinline = 0;
  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = mol->segment_array[iseg];
    if (! seg) continue;
    nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      res = &(seg->residue_array[ires]);
      for ( atom = res->atoms; atom; atom = atom->next ) {
        for ( dihe = atom->dihedrals; dihe;
                dihe = topo_mol_dihedral_next(dihe,atom) ) {
          if ( dihe->atom[0] == atom && ! dihe->del ) {
            if ( numinline == 2 ) { fprintf(file,"\n");  numinline = 0; }
            fprintf(file, ( charmmext ? " %9d %9d %9d %9d" : " %7d %7d %7d %7d"),atom->atomid,
                dihe->atom[1]->atomid,dihe->atom[2]->atomid,
                dihe->atom[3]->atomid);
            ++numinline;
          }
        }
      }
    }
  }
  fprintf(file,"\n\n");

  fprintf(file,"%8d !NIMPHI: impropers\n",nimprs);
  numinline = 0;
  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = mol->segment_array[iseg];
    if (! seg) continue;
    nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      res = &(seg->residue_array[ires]);
      for ( atom = res->atoms; atom; atom = atom->next ) {
        for ( impr = atom->impropers; impr;
                impr = topo_mol_improper_next(impr,atom) ) {
          if ( impr->atom[0] == atom && ! impr->del ) {
            if ( numinline == 2 ) { fprintf(file,"\n");  numinline = 0; }
            fprintf(file, ( charmmext ? " %9d %9d %9d %9d" : " %7d %7d %7d %7d"),atom->atomid,
                impr->atom[1]->atomid,impr->atom[2]->atomid,
                impr->atom[3]->atomid);
            ++numinline;
          }
        }
      }
    }
  }
  fprintf(file,"\n\n");

  fprintf(file,"%8d !NDON: donors\n\n\n",0);
  fprintf(file,"%8d !NACC: acceptors\n\n\n",0);
  fprintf(file,"%8d !NNB\n\n",0);
  /* Pad with zeros, one for every atom */
  {
    int i, fullrows;
    fullrows = atomid/8;
    for (i=0; i<fullrows; ++i) 
      fprintf(file, (charmmext?"%10d%10d%10d%10d%10d%10d%10d%10d\n":"%8d%8d%8d%8d%8d%8d%8d%8d\n"),0,0,0,0,0,0,0,0);
    for (i=atomid - fullrows*8; i; --i)
      fprintf(file, (charmmext?"%10d":"%8d"),0);
  } 
  fprintf(file,"\n\n");

  fprintf(file,(charmmext?"%8d %7d !NGRP\n%10d%10d%10d\n\n":"%8d %7d !NGRP\n%8d%8d%8d\n\n"),1,0,0,0,0);

  if ( ! nocmap ) {
    fprintf(file,"%8d !NCRTERM: cross-terms\n",ncmaps);
    for ( iseg=0; iseg<nseg; ++iseg ) {
      seg = mol->segment_array[iseg];
      if (! seg) continue;
      nres = hasharray_count(seg->residue_hash);
      for ( ires=0; ires<nres; ++ires ) {
        res = &(seg->residue_array[ires]);
        for ( atom = res->atoms; atom; atom = atom->next ) {
          for ( cmap = atom->cmaps; cmap;
                  cmap = topo_mol_cmap_next(cmap,atom) ) {
            if ( cmap->atom[0] == atom && ! cmap->del ) {
              fprintf(file,( charmmext ? " %9d %9d %9d %9d %9d %9d %9d %9d\n"
                         : " %7d %7d %7d %7d %7d %7d %7d %7d\n"),atom->atomid,
                  cmap->atom[1]->atomid,cmap->atom[2]->atomid,
                  cmap->atom[3]->atomid,cmap->atom[4]->atomid,
                  cmap->atom[5]->atomid,cmap->atom[6]->atomid,
                  cmap->atom[7]->atomid);
            }
          }
        }
      }
    }
    fprintf(file,"\n");
  }

  return 0;
}

