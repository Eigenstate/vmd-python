#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "pdb_file_extract.h"
#include "pdb_file.h"
#include "extract_alias.h"

#if defined(_MSC_VER)
#define snprintf _snprintf
#endif

static void strtoupper(char *s) {
  while ( *s ) { *s = toupper(*s); ++s; }
}

int pdb_file_extract_residues(topo_mol *mol, FILE *file, stringhash *h, int all_caps,
                                void *v,void (*print_msg)(void *,const char *)) {

  char record[PDB_RECORD_LENGTH+2];
  int indx;
  float x,y,z,o,b;
  char name[8], resname[8], chain[8];
  char segname[8], element[8], resid[8], insertion[8];
  char oldresid[8];
  const char *realres;
  char msg[128];
  int rcount;

  rcount = 0;
  oldresid[0] = '\0';

  do {
    if((indx = read_pdb_record(file, record)) == PDB_ATOM) {
      get_pdb_fields(record, name, resname, chain,
                   segname, element, resid, insertion, &x, &y, &z, &o, &b);
      if ( strcmp(oldresid,resid) ) {
        strcpy(oldresid,resid);
        ++rcount;
        if ( all_caps ) strtoupper(resname);
        if ( all_caps ) strtoupper(chain);
        realres = extract_alias_residue_check(h,resname);
        if ( topo_mol_residue(mol,resid,realres,chain) ) {
          sprintf(msg,"ERROR: failed on residue %s from pdb file",resname);
          print_msg(v,msg);
        }
      }
    }
  } while (indx != PDB_END && indx != PDB_EOF);

  sprintf(msg,"extracted %d residues from pdb file",rcount);
  print_msg(v,msg);
  return 0;
}

int pdb_file_extract_coordinates(topo_mol *mol, FILE *file, FILE *namdbinfile,
                                const char *segid, stringhash *h, int all_caps,
                                void *v,void (*print_msg)(void *,const char *)) {

  char record[PDB_RECORD_LENGTH+2];
  int indx;
  topo_mol_ident_t target;
  char msg[128];
  unsigned int utmp;
  char stmp[128];

  int numatoms, pdbnatoms;
  double *atomcoords = 0;

  if ( namdbinfile ) {
    char static_assert_int_is_32_bits[sizeof(int) == 4 ? 1 : -1];
    int filen;
    int wrongendian;

    fseek(namdbinfile,0,SEEK_END);
    numatoms = (ftell(namdbinfile)-4)/24;
    if (numatoms < 1) {
      print_msg(v,"namdbin file is too short");
      return -1;
    }
    fseek(namdbinfile,0,SEEK_SET);
    fread(&filen, sizeof(int), 1, namdbinfile);
    wrongendian = 0;
    if (filen != numatoms) {
      char lenbuf[4];
      char tmpc;
      wrongendian = 1;
      memcpy(lenbuf, (const char *)&filen, 4);
      tmpc = lenbuf[0]; lenbuf[0] = lenbuf[3]; lenbuf[3] = tmpc;
      tmpc = lenbuf[1]; lenbuf[1] = lenbuf[2]; lenbuf[2] = tmpc;
      memcpy((char *)&filen, lenbuf, 4);
    }
    if (filen != numatoms) {
      print_msg(v,"inconsistent atom count in namdbin file");
      return -1;
    }
    if (wrongendian) {
      print_msg(v,"namdbin file appears to be other-endian");
    }
    atomcoords = (double *)malloc(numatoms * 3 * sizeof(double));
    if (fread(atomcoords, sizeof(double), 3 * numatoms, namdbinfile)
                                 != (size_t)(3 * numatoms)) {
      print_msg(v,"error reading data from namdbin file");
      free(atomcoords);
      return -1;
    }
    if (wrongendian) {
      int i;
      char tmp0, tmp1, tmp2, tmp3;
      char *cdata = (char *) atomcoords;
      print_msg(v,"converting other-endian data from namdbin file");
      for ( i=0; i<3*numatoms; ++i, cdata+=8 ) {
        tmp0 = cdata[0]; tmp1 = cdata[1];
        tmp2 = cdata[2]; tmp3 = cdata[3];
        cdata[0] = cdata[7]; cdata[1] = cdata[6];
        cdata[2] = cdata[5]; cdata[3] = cdata[4];
        cdata[7] = tmp0; cdata[6] = tmp1;
        cdata[5] = tmp2; cdata[4] = tmp3;
      }
    }
  }

  target.segid = segid;
  pdbnatoms = 0;

  do {
    if((indx = read_pdb_record(file, record)) == PDB_ATOM) {
      float xf,yf,zf,o,b;
      double x,y,z;
      char name[8], altname[8], resname[8], chain[8];
      char segname[8], element[8], resid[8], insertion[8];
      int found;
      get_pdb_fields(record, name, resname, chain,
                   segname, element, resid, insertion, &xf, &yf, &zf, &o, &b);
      x = xf;  y=yf;  z=zf;
      if ( namdbinfile ) {
        if (pdbnatoms >= numatoms) {
          print_msg(v,"too few atoms in namdbin file");
          free(atomcoords);
          return -1;
        }
        x = atomcoords[pdbnatoms*3    ];
        y = atomcoords[pdbnatoms*3 + 1];
        z = atomcoords[pdbnatoms*3 + 2];
        ++pdbnatoms;
      }
      target.resid = resid;
      if ( all_caps ) strtoupper(resname);
      if ( all_caps ) strtoupper(name);
      if ( all_caps ) strtoupper(chain);
      target.aname = extract_alias_atom_check(h,resname,name);
      /* Use PDB segid if no segid given */
      if (!segid) {
        target.segid = segname;
      }
      found = ! topo_mol_set_xyz(mol,&target,x,y,z);
      /* Try reversing order so 1HE2 in pdb matches HE21 in topology */
      if ( ! found && sscanf(name,"%u%s",&utmp,stmp) == 2 ) {
        snprintf(altname,8,"%s%u",stmp,utmp);
        target.aname = altname;
        if ( ! topo_mol_set_xyz(mol,&target,x,y,z) ) {
          found = 1;
          /*  too much information
          sprintf(msg,"Warning: changed atom name for atom %s\t %s:%s\t  %s to %s",name,resname,resid,segid ? segid : segname,altname);
          print_msg(v,msg);
          */
        }
      }
      if ( ! found ) {
        sprintf(msg,"Warning: failed to set coordinate for atom %s\t %s:%s\t  %s",name,resname,resid,segid ? segid : segname);
        print_msg(v,msg);
      } else {
        /* only try element and chain if coordinates succeeds */
        if ( strlen(element) && topo_mol_set_element(mol,&target,element,0) ) {
          sprintf(msg,"Warning: failed to set element for atom %s\t %s:%s\t  %s",name,resname,resid,segid ? segid : segname);
          print_msg(v,msg);
        }
        if ( strlen(chain) && topo_mol_set_chain(mol,&target,chain,0) ) {
          sprintf(msg,"Warning: failed to set chain for atom %s\t %s:%s\t  %s",name,resname,resid,segid ? segid : segname);
          print_msg(v,msg);
        }
      }
    }
  } while (indx != PDB_END && indx != PDB_EOF);

  if ( namdbinfile ) {
    free(atomcoords);
    if (numatoms > pdbnatoms) {
      print_msg(v,"too many atoms in namdbin file");
      return -1;
    }
  }

  return 0;

}


