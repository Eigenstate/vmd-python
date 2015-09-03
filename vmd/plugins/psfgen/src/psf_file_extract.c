#include <stdlib.h>
#include <string.h>
#include "psf_file.h"
#include "psf_file_extract.h"
#include "pdb_file.h"
#include "topo_mol_struct.h"

/* General note: in a few places I read various arrays in reverse order. 
   That's because I want psf files emitted by psfgen to have all the atoms,
   bonds, etc. in the same order as in the original psf file.  We have to 
   reverse it because adding to a linked list reverses the order.  Actually,
   if the original psf file comes from some other program, then we might 
   change the order of the bonds, angles, etc. but we can at least guarantee
   that if we read a psf file written by psfgen, then write it out again,
   the output will match the input exactly.
*/

/* Read in all psf atom information using this struct */
struct psfatom {
  char name[10];
  char atype[10];
  char resname[10];
  char segname[10];
  char resid[10];
  char element[3];
  double charge, mass;
};
typedef struct psfatom psfatom;

#define PSF_RECORD_LENGTH 	200


static int extract_patches(FILE *file, topo_mol *mol) { 
  char inbuf[PSF_RECORD_LENGTH+2];
  int npatch = 0;
  
  /* Read comments; get patch info */
  while (!feof(file)) {
    if (inbuf != fgets(inbuf, PSF_RECORD_LENGTH+1, file)) {
      /* EOF with no NATOM */
      return -1;  
    }
    if (strlen(inbuf) > 0) {
      if (strstr(inbuf, "REMARKS")) {
	char *pbuf;
	if (strstr(inbuf, "REMARKS topology ")) {
	  char topofile[256];
	  pbuf = strstr(inbuf, "topology");
	  pbuf=pbuf+strlen("topology");
	  sscanf(pbuf, "%s", topofile);	  
	  topo_defs_add_topofile(mol->defs, topofile);
	}
	if (strstr(inbuf, "REMARKS patch ") || strstr(inbuf, "REMARKS defaultpatch ")) {
	  char pres[NAMEMAXLEN], segres[2*NAMEMAXLEN];
	  char s[NAMEMAXLEN], r[NAMEMAXLEN];
	  topo_mol_ident_t target;
	  pbuf = strstr(inbuf, "patch");
	  pbuf=pbuf+5;
	  sscanf(pbuf, "%s", pres);
	  if (strcmp(pres,"----")) {
	    if (strstr(inbuf, "REMARKS defaultpatch")) {
	      topo_mol_add_patch(mol,pres,1);
	    } else {
	      topo_mol_add_patch(mol,pres,0);
	    }
	  }
	  pbuf = strstr(pbuf, pres)+strlen(pres);
	  while (sscanf(pbuf, "%s", segres)==1) { 
	    int slen;
	    slen = strcspn(segres,":");
	    strncpy(s, segres, slen);
	    s[slen] = '\0';
	    strcpy(r, strchr(segres,':')+1);
	    target.segid = s;
	    target.resid = r;
	    topo_mol_add_patchres(mol,&target);
	    pbuf = strstr(pbuf,segres)+strlen(segres);
	  }
	  npatch++;
	}
      } else {
	if (strstr(inbuf, "NATOM")) {
          rewind(file);
	  return npatch;
        }
      }
    }
  } ;
  return npatch;
}


static int extract_segment_extra_data(FILE *file, topo_mol *mol) {
  char inbuf[PSF_RECORD_LENGTH+2];
  
  /* Read comments; get patch info */
  while (!feof(file)) {
    if (inbuf != fgets(inbuf, PSF_RECORD_LENGTH+1, file)) {
      /* EOF with no NATOM */
      return -1;  
    }
    if (strlen(inbuf) > 0) {
      if (strstr(inbuf, "REMARKS")) {
	char *pbuf;
	if (strstr(inbuf, "REMARKS segment ")) {
	  char segid[NAMEMAXLEN], pfirst[NAMEMAXLEN], plast[NAMEMAXLEN];
	  char angles[20], diheds[20], tmp[NAMEMAXLEN];
	  topo_mol_segment_t *seg = NULL;
	  int id;
	  pbuf = strstr(inbuf, "segment");
	  pbuf += strlen("segment");
	  sscanf(pbuf, "%s %s %s %s %s %s %s %s %s", segid, tmp, tmp, pfirst, tmp, plast, tmp, angles, diheds);
	  if ( (id = hasharray_index(mol->segment_hash, segid)) != HASHARRAY_FAIL) {
	    /* Then the segment exists.  Look it up and return it. */
	    seg = mol->segment_array[id];
	    strcpy(strchr(pfirst,';'),"");
	    strcpy(strchr(plast, ';'),"");
	    strcpy(seg->pfirst,pfirst);
	    strcpy(seg->plast, plast);
	    seg->auto_angles = 0; 
	    if (!strcmp(angles,"angles")) {
	      seg->auto_angles = 1; 
	    }
	    seg->auto_dihedrals = 0; 
	    if (!strcmp(diheds,"dihedrals")) {
	      seg->auto_dihedrals = 1; 
	    }
	  } 
	}
      }
    }
  }
  return 0;
}

static int extract_bonds(FILE *file, int fw, topo_mol *mol, int natoms, 
                         topo_mol_atom_t **molatomlist) {

  int *bonds;
  int i, nbonds;

  /* Build bonds */
  nbonds = psf_start_block(file, "NBOND");
  if (nbonds < 0) {
    return -1; 
  }
  bonds = (int *)malloc(2*nbonds*sizeof(int));

  if (psf_get_bonds(file, fw, nbonds, bonds)) {
    free(bonds);
    return -1;
  }
 
  for (i=nbonds-1; i >= 0; i--) {
    topo_mol_atom_t *atom1, *atom2;
    topo_mol_bond_t *tuple;
    int ind1, ind2; 
  
    ind1 = bonds[2*i]-1; 
    ind2 = bonds[2*i+1]-1;
    if (ind1 < 0 || ind2 < 0 || ind1 >= natoms || ind2 >= natoms) {
      /* Bad indices, abort now */
      free(bonds);
      return -1;
    }
    
    atom1 = molatomlist[ind1];
    atom2 = molatomlist[ind2];
   
    tuple = memarena_alloc(mol->arena,sizeof(topo_mol_bond_t));
    tuple->next[0] = atom1->bonds;
    tuple->atom[0] = atom1;
    tuple->next[1] = atom2->bonds;
    tuple->atom[1] = atom2;
    tuple->del = 0;
 
    atom1->bonds = tuple; 
    atom2->bonds = tuple;
  }
  free(bonds);
  return 0;
}

static int extract_angles(FILE *file, int fw, topo_mol *mol, int natoms, 
                         topo_mol_atom_t **molatomlist) {

  int i, nangles;
  int *angles;
  
  nangles = psf_start_block(file, "NTHETA");
  if (nangles < 0) return -1; 
  angles = (int *)malloc(3*nangles*sizeof(int));

  if (psf_get_angles(file, fw, nangles, angles)) {
    free(angles); 
    return -1; 
  } 
  
  for (i=nangles-1; i >= 0; i--) {
    topo_mol_atom_t *atom1, *atom2, *atom3;
    topo_mol_angle_t *tuple;

    atom1 = molatomlist[angles[3*i]-1];
    atom2 = molatomlist[angles[3*i+1]-1];
    atom3 = molatomlist[angles[3*i+2]-1];
   
    tuple = memarena_alloc(mol->angle_arena,sizeof(topo_mol_angle_t));
    tuple->next[0] = atom1->angles;
    tuple->atom[0] = atom1;
    tuple->next[1] = atom2->angles;
    tuple->atom[1] = atom2;
    tuple->next[2] = atom3->angles;
    tuple->atom[2] = atom3;
    tuple->del = 0;
 
    atom1->angles = tuple; 
    atom2->angles = tuple;
    atom3->angles = tuple;
  }
  free(angles);
  return 0;
}

static int extract_dihedrals(FILE *file, int fw, topo_mol *mol, int natoms,
                         topo_mol_atom_t **molatomlist) {

  int i, ndihedrals;
  int *dihedrals;

  ndihedrals = psf_start_block(file, "NPHI");
  if (ndihedrals < 0) return -1; 
  dihedrals = (int *)malloc(4*ndihedrals*sizeof(int));

  if (psf_get_dihedrals(file, fw, ndihedrals, dihedrals)) {
    free(dihedrals); 
    return -1;
  }
   
  for (i=ndihedrals-1; i >= 0; i--) {
    topo_mol_atom_t *atom1, *atom2, *atom3, *atom4;
    topo_mol_dihedral_t *tuple;

    atom1 = molatomlist[dihedrals[4*i]-1];
    atom2 = molatomlist[dihedrals[4*i+1]-1];
    atom3 = molatomlist[dihedrals[4*i+2]-1];
    atom4 = molatomlist[dihedrals[4*i+3]-1];

    tuple = memarena_alloc(mol->dihedral_arena,sizeof(topo_mol_dihedral_t));
    tuple->next[0] = atom1->dihedrals;
    tuple->atom[0] = atom1;
    tuple->next[1] = atom2->dihedrals;
    tuple->atom[1] = atom2;
    tuple->next[2] = atom3->dihedrals;
    tuple->atom[2] = atom3;
    tuple->next[3] = atom4->dihedrals;
    tuple->atom[3] = atom4;
    tuple->del = 0;

    atom1->dihedrals = tuple;
    atom2->dihedrals = tuple;
    atom3->dihedrals = tuple;
    atom4->dihedrals = tuple;
  }
  free(dihedrals);
  return 0;
}

static int extract_impropers(FILE *file, int fw, topo_mol *mol, int natoms,
                         topo_mol_atom_t **molatomlist) {

  int i, nimpropers;
  int *impropers;
  
  nimpropers = psf_start_block(file, "NIMPHI");
  if (nimpropers < 0) return -1; 
  impropers = (int *)malloc(4*nimpropers*sizeof(int));

  if (psf_get_impropers(file, fw, nimpropers, impropers)) {
    free(impropers); 
    return -1;
  } 
    
  for (i=nimpropers-1; i >= 0; i--) {
    topo_mol_atom_t *atom1, *atom2, *atom3, *atom4;
    topo_mol_improper_t *tuple;

    atom1 = molatomlist[impropers[4*i]-1];
    atom2 = molatomlist[impropers[4*i+1]-1];
    atom3 = molatomlist[impropers[4*i+2]-1];
    atom4 = molatomlist[impropers[4*i+3]-1];
   
    tuple = memarena_alloc(mol->arena,sizeof(topo_mol_improper_t));
    tuple->next[0] = atom1->impropers;
    tuple->atom[0] = atom1;
    tuple->next[1] = atom2->impropers;
    tuple->atom[1] = atom2;
    tuple->next[2] = atom3->impropers;
    tuple->atom[2] = atom3;
    tuple->next[3] = atom4->impropers;
    tuple->atom[3] = atom4;
    tuple->del = 0;
 
    atom1->impropers = tuple; 
    atom2->impropers = tuple;
    atom3->impropers = tuple;
    atom4->impropers = tuple;
  }
  free(impropers);
  return 0;
}

static int extract_cmaps(FILE *file, int fw, topo_mol *mol, int natoms,
                         topo_mol_atom_t **molatomlist) {

  int i, j, ncmaps;
  int *cmaps;
  
  ncmaps = psf_start_block(file, "NCRTERM");
  if (ncmaps < 0) {
    return 1;
  }
  cmaps = (int *)malloc(8*ncmaps*sizeof(int));

  if (psf_get_cmaps(file, fw, ncmaps, cmaps)) {
    free(cmaps); 
    return -1;
  } 
    
  for (i=ncmaps-1; i >= 0; i--) {
    topo_mol_atom_t *atoml[8];
    topo_mol_cmap_t *tuple;

    tuple = memarena_alloc(mol->arena,sizeof(topo_mol_cmap_t));
    for ( j = 0; j < 8; ++j ) {
      atoml[j] = molatomlist[cmaps[8*i+j]-1];
      tuple->next[j] = atoml[j]->cmaps;
      tuple->atom[j] = atoml[j];
    }
    tuple->del = 0;
    for ( j = 0; j < 8; ++j ) {
      atoml[j]->cmaps = tuple; 
    }
  }

  free(cmaps);
  return 0;
}

/* Return the segment corresponding to the given segname.  If the segname
   doesn't exist, add it.  Return NULL on error.
*/
static topo_mol_segment_t *get_segment(topo_mol *mol, const char *segname) {
  int id;
  topo_mol_segment_t *seg = NULL;
  
  if ( (id = hasharray_index(mol->segment_hash, segname)) != HASHARRAY_FAIL) {
    /* Then the segment exists.  Look it up and return it. */
    seg = mol->segment_array[id];
  } else {
    /* Must create new segment */
    id = hasharray_insert(mol->segment_hash, segname);
    if (id != HASHARRAY_FAIL) {
      seg = mol->segment_array[id] =
            (topo_mol_segment_t *) malloc(sizeof(topo_mol_segment_t)); 
      strcpy(seg->segid, segname);
      seg->residue_hash = hasharray_create(
        (void**) &(seg->residue_array), sizeof(topo_mol_residue_t));
      strcpy(seg->pfirst,"");
      strcpy(seg->plast,"");
      seg->auto_angles = 0; 
      seg->auto_dihedrals = 0; 
    }
  }
  return seg;
}

/* Return a new residue with the given resid.  Add it to the given segment.
   If the resid already exists, return NULL.  Return NULL if there's a problem.
*/

static topo_mol_residue_t *get_residue(topo_mol_segment_t *seg, 
        const char *resid) {
  
  int id;
  topo_mol_residue_t *res;
  
  /* Check that the residue doesn't already exist */
  if ( hasharray_index(seg->residue_hash,resid) != HASHARRAY_FAIL ) {
    return NULL; 
  }
  id = hasharray_insert(seg->residue_hash, resid);
  if (id == HASHARRAY_FAIL) {
    return NULL;
  }
  res = &(seg->residue_array[id]);
  strcpy(res->resid, resid);
  
  return res;
}


int psf_file_extract(topo_mol *mol, FILE *file, FILE *pdbfile, FILE *namdbinfile, void *v,
                                void (*print_msg)(void *, const char *)) {
  int i, natoms, npatch, charmmext;
  psfatom *atomlist;
  double *atomcoords;
  topo_mol_atom_t **molatomlist;
  long filepos;
  char inbuf[PSF_RECORD_LENGTH+2];

  /* Read header flags */
  if (feof(file) || (inbuf != fgets(inbuf, PSF_RECORD_LENGTH+1, file))) {
    print_msg(v,"ERROR: Unable to read psf file");
    return -1;
  }
  if ( strncmp(inbuf, "PSF", 3) ) {
    print_msg(v,"ERROR: File does not begin with PSF - wrong format?");
    return -1;
  }
  charmmext = ( strstr(inbuf, "EXT") ? 1 : 0 ); 

  /* Read patch info from REMARKS */
  npatch = extract_patches(file, mol);

  natoms = psf_start_atoms(file);
  if (natoms < 0) {
    print_msg(v,"ERROR: Unable to read psf file");
    return -1;
  }
 
  atomlist = (psfatom *)malloc(natoms * sizeof(psfatom));

  /* Read in all atoms */
  for (i=0; i<natoms; i++) {
    psfatom *atom = atomlist + i;
    strcpy(atom->element,"");
    if (psf_get_atom(file, atom->name,atom->atype,atom->resname, atom->segname,
                     atom->resid, &atom->charge, &atom->mass)
        < 0) {
      print_msg(v,"error reading atoms from psf file");
      free(atomlist);
      return -1;
    }
  }

  /* Optionally read coordinates, insertion code, and element symbol from PDB file */
  atomcoords = 0;
  if ( pdbfile ) {
    char record[PDB_RECORD_LENGTH+2];
    int indx, insertions;
    float x,y,z,o,b;
    char name[8], resname[8], chain[8];
    char segname[8], element[8], resid[8], insertion[8];

    atomcoords = (double *)malloc(natoms * 3 * sizeof(double));

    insertions = 0;
    i=0; 
    do {
      if((indx = read_pdb_record(pdbfile, record)) == PDB_ATOM) {
        psfatom *atom = atomlist + i;
        if ( i >= natoms ) {
          print_msg(v,"too many atoms in pdb file");
          free(atomlist);
          free(atomcoords);
          return -1;
        }
        get_pdb_fields(record, name, resname, chain,
                   segname, element, resid, insertion, &x, &y, &z, &o, &b);
        if ( strncmp(atom->name,name,4) ||
             strncmp(atom->resname,resname,4) ||
             strncmp(atom->segname,segname,4) ) {
          print_msg(v,"atom mismatch in pdb file");
          print_msg(v,record);
          free(atomlist);
          free(atomcoords);
          return -1;
        }
        if ( insertion[0] != ' ' && insertion[0] != '\0' ) {
          if ( ( strlen(atom->resid ) <= 4 ) && strncmp(atom->resid,resid,4) ) {
            strncpy(atom->resid,resid,7);  atom->resid[7] = 0;
            ++insertions;
          }
          if ( atom->resid[strlen(atom->resid)-1] != insertion[0] ) {
            strncat(atom->resid,insertion,1);
          }
        }
        strncpy(atom->element,element,3);  atom->element[2] = 0;
        atomcoords[i*3    ] = x;
        atomcoords[i*3 + 1] = y;
        atomcoords[i*3 + 2] = z;
        ++i;
      }
    } while (indx != PDB_END && indx != PDB_EOF);
    if ( insertions ) {
      char buf[80];
      sprintf(buf, "Found %d mismatched resids with insertion codes in pdb file", insertions);
      print_msg(v,buf);
    }
    if ( i < natoms ) {
      print_msg(v,"too few atoms in pdb file");
      free(atomlist);
      free(atomcoords);
      return -1;
    }
  }

  if ( namdbinfile ) {
    int numatoms;
    int filen;
    int wrongendian;
    char lenbuf[4];
    char tmpc;
    char static_assert_int_is_32_bits[sizeof(int) == 4 ? 1 : -1];
    if ( ! atomcoords ) {
      atomcoords = (double *)malloc(natoms * 3 * sizeof(double));
    }
    fseek(namdbinfile,0,SEEK_END);
    numatoms = (ftell(namdbinfile)-4)/24;
    if (numatoms < 1) {
      print_msg(v,"namdbin file is too short");
      free(atomlist);
      free(atomcoords);
      return -1;
    }
    fseek(namdbinfile,0,SEEK_SET);
    fread(&filen, sizeof(int), 1, namdbinfile);
    wrongendian = 0;
    if (filen != numatoms) {
      wrongendian = 1;
      memcpy(lenbuf, (const char *)&filen, 4);
      tmpc = lenbuf[0]; lenbuf[0] = lenbuf[3]; lenbuf[3] = tmpc;
      tmpc = lenbuf[1]; lenbuf[1] = lenbuf[2]; lenbuf[2] = tmpc;
      memcpy((char *)&filen, lenbuf, 4);
    }
    if (filen != numatoms) {
      print_msg(v,"inconsistent atom count in namdbin file");
      free(atomlist);
      free(atomcoords);
      return -1;
    }
    if (numatoms < natoms) {
      print_msg(v,"too few atoms in namdbin file");
      free(atomlist);
      free(atomcoords);
      return -1;
    }
    if (numatoms > natoms) {
      print_msg(v,"too many atoms in namdbin file");
      free(atomlist);
      free(atomcoords);
      return -1;
    }
    if (wrongendian) {
      print_msg(v,"namdbin file appears to be other-endian");
    }
    if (fread(atomcoords, sizeof(double), 3 * natoms, namdbinfile)
                                 != (size_t)(3 * natoms)) {
      print_msg(v,"error reading data from namdbin file");
      free(atomlist);
      free(atomcoords);
      return -1;
    }
    if (wrongendian) {
      int i;
      char tmp0, tmp1, tmp2, tmp3;
      char *cdata = (char *) atomcoords;
      print_msg(v,"converting other-endian data from namdbin file");
      for ( i=0; i<3*natoms; ++i, cdata+=8 ) {
        tmp0 = cdata[0]; tmp1 = cdata[1];
        tmp2 = cdata[2]; tmp3 = cdata[3];
        cdata[0] = cdata[7]; cdata[1] = cdata[6];
        cdata[2] = cdata[5]; cdata[3] = cdata[4];
        cdata[7] = tmp0; cdata[6] = tmp1;
        cdata[5] = tmp2; cdata[4] = tmp3;
      }
    }
  }
 
  molatomlist = (topo_mol_atom_t **)malloc(natoms * sizeof(topo_mol_atom_t *));

  i=0; 
  while (i < natoms) {
    topo_mol_segment_t *seg;
    topo_mol_residue_t *res;
    topo_mol_atom_t *atomtmp;
    int firstatom, j;
    const char *resid, *segname;

    resid = atomlist[i].resid;
    segname = atomlist[i].segname;
    seg = get_segment(mol, segname);
    if (!seg) { 
      print_msg(v,"ERROR: unable to get segment!");
      break;
    }
    res = get_residue(seg, resid);
    if (!res) {
      char *buf;
      int len = strlen(resid) + strlen(segname);
      buf = (char *)malloc((50 + len)*sizeof(char));
      sprintf(buf, "Unable to add (duplicate?) residue %s:%s", segname, resid);
      print_msg(v,buf);
      free(buf);
      break;
    }
    strcpy(res->name, atomlist[i].resname);
    strcpy(res->chain, "");
    res->atoms = 0;
    firstatom = i;
    while (i<natoms && !strcmp(resid, atomlist[i].resid) &&
                       !strcmp(segname, atomlist[i].segname)) {
      /* Add atoms to residue */
      atomtmp = memarena_alloc(mol->arena, sizeof(topo_mol_atom_t));
      atomtmp->bonds = 0;
      atomtmp->angles = 0;
      atomtmp->dihedrals = 0;
      atomtmp->impropers = 0;
      atomtmp->cmaps = 0;
      atomtmp->conformations = 0;
      strcpy(atomtmp->name, atomlist[i].name);
      strcpy(atomtmp->type, atomlist[i].atype);
      strcpy(atomtmp->element, atomlist[i].element);
      atomtmp->mass = atomlist[i].mass; 
      atomtmp->charge = atomlist[i].charge;
      if (atomcoords) {
        atomtmp->x = atomcoords[i*3    ];
        atomtmp->y = atomcoords[i*3 + 1];
        atomtmp->z = atomcoords[i*3 + 2];
        atomtmp->xyz_state = TOPO_MOL_XYZ_SET;
      } else {
        atomtmp->x = 0;       
        atomtmp->y = 0;       
        atomtmp->z = 0;       
        atomtmp->xyz_state = TOPO_MOL_XYZ_VOID;
      }
      atomtmp->partition = 0;
      atomtmp->copy = 0;
      atomtmp->atomid = 0;

      /* Save pointer to atom in my table so I can put in the bond 
         information without having find the atom.
      */
      molatomlist[i] = atomtmp;
      i++;
    }
    for (j=i-1; j >= firstatom; j--) {
      /* Add new atoms to head of linked list in reverse order, so that
         the linked list is in the order they appear in the psf file. 
      */
      atomtmp = molatomlist[j];
      atomtmp->next = res->atoms;
      res->atoms = atomtmp;
    }  
  }  

  if (atomcoords) free(atomcoords);
  atomcoords = 0;

  /* Check to see if we broke out of the loop prematurely */
  if (i != natoms) {
    free(atomlist);
    free(molatomlist);
    return -1;
  }

  /* Get the segment patch first,last and auto angles,dihedrals info from psf */
  /* We have to rewind the file and read the info now since it has to be added to */
  /* the existing segments which have just been read. */
  filepos = ftell(file);
  rewind(file);
  extract_segment_extra_data(file, mol);
  fseek(file, filepos, SEEK_SET);

  if (extract_bonds(file, (charmmext ? 10 : 8), mol, natoms, molatomlist)) {
    print_msg(v,"Error processing bonds");
    free(atomlist);
    free(molatomlist);
    return -1;
  }
 
  if (extract_angles(file, (charmmext ? 10 : 8), mol, natoms, molatomlist)) {
    print_msg(v,"Error processing angles");
    free(atomlist);
    free(molatomlist);
    return -1;
  }

  if (extract_dihedrals(file, (charmmext ? 10 : 8), mol, natoms, molatomlist)) {
    print_msg(v,"Error processing dihedrals");
    free(atomlist);
    free(molatomlist);
    return -1;
  }

  if (extract_impropers(file, (charmmext ? 10 : 8), mol, natoms, molatomlist)) {
    print_msg(v,"Error processing impropers");
    free(atomlist);
    free(molatomlist);
    return -1;
  }

  switch (extract_cmaps(file, (charmmext ? 10 : 8), mol, natoms, molatomlist)) {
  case 0:
    break;
  case 1:
    print_msg(v,"psf file does not contain cross-terms");
    break;
  default:
    print_msg(v,"Error processing cross-terms");
    free(atomlist);
    free(molatomlist);
    return -1;
  }

  free(atomlist);
  free(molatomlist);
  return 0;
}


