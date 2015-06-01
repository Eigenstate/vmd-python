 
/***************************************************************************
 * DESCRIPTION:
 *
 * General routines to read .pdb files.
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pdb_file.h"

/* read the next record from the specified pdb file, and put the string found
   in the given string pointer (the caller must provide adequate (81 chars)
   buffer space); return the type of record found
*/
int read_pdb_record(FILE *f, char *retStr) {

  char inbuf[PDB_RECORD_LENGTH+2];
  int recType = PDB_UNKNOWN;
  
  /*	read the next line	*/
  if(inbuf != fgets(inbuf, PDB_RECORD_LENGTH+1, f)) {
    strcpy(retStr,"");
    recType = PDB_EOF;
  } else {
    /*	remove the newline character, if there is one */
    if(inbuf[strlen(inbuf)-1] == '\n')
      inbuf[strlen(inbuf)-1] = '\0';

    /* what was it? */
    if (!strncmp(inbuf, "REMARK", 6)) {
      recType = PDB_REMARK;

    } else if (!strncmp(inbuf, "CRYST1", 6)) {
      recType = PDB_CRYST1;
      
    } else if (!strncmp(inbuf, "ATOM  ", 6) ||
	       !strncmp(inbuf, "HETATM", 6)) {
      recType = PDB_ATOM;

      /* the only two END records are "END   " and "ENDMDL" */
    } else if (!strcmp(inbuf, "END") ||       /* If not space " " filled */
	       !strncmp(inbuf, "END ", 4) ||  /* Allows other stuff */
	       !strncmp(inbuf, "ENDMDL", 6)) { /* NMR records */
      recType = PDB_END;

    } else {
      recType = PDB_UNKNOWN;
    }

    if(recType == PDB_REMARK || recType == PDB_ATOM || 
       recType == PDB_CRYST1) {
      strcpy(retStr,inbuf);
    } else {
      strcpy(retStr,"");
    }
  }

  /* read the '\r', if there was one */
  {
    int ch = fgetc(f);
    if (ch != '\r') {
      ungetc(ch, f);
    }
  }
  
  return recType;
}

void get_pdb_cryst1(char *record, float *alpha, float *beta, float *gamma,
		    float *a, float *b, float *c)
{
  char tmp[81];
  char ch, *s;
  int i;
  for (i=0; i<81; i++) tmp[i] = 0;
  strncpy(tmp, record, 80);
  tmp[80] = 0;
  s = tmp;

  s = tmp+6 ;          ch = tmp[15]; tmp[15] = 0;
  *a = (float) atof(s);
  s = tmp+15; *s = ch; ch = tmp[24]; tmp[24] = 0;
  *b = (float) atof(s);
  s = tmp+24; *s = ch; ch = tmp[33]; tmp[33] = 0;
  *c = (float) atof(s);
  s = tmp+33; *s = ch; ch = tmp[40]; tmp[40] = 0;
  *alpha = (float) atof(s);
  s = tmp+40; *s = ch; ch = tmp[47]; tmp[47] = 0;
  *beta = (float) atof(s);
  s = tmp+47; *s = ch; ch = tmp[54]; tmp[54] = 0;
  *gamma = (float) atof(s);
}

/* Extract the x,y, and z coordinates from the given ATOM record.	*/
void get_pdb_coordinates(char *record, float *x, float *y, float *z,
	float *occup, float *beta) {
  char numstr[9];

  /* get X, Y, and Z */
  memset(numstr, 0, 9 * sizeof(char));
  strncpy(numstr, record + 30, 8);
  *x = (float) atof(numstr);
  memset(numstr, 0, 9 * sizeof(char));
  strncpy(numstr, record + 38, 8);
  *y = (float) atof(numstr);
  memset(numstr, 0, 9 * sizeof(char));
  strncpy(numstr, record + 46, 8);
  *z = (float) atof(numstr);
  memset(numstr, 0, 9 * sizeof(char));
  strncpy(numstr, record + 54, 6);
  *occup = (float) atof(numstr);
  memset(numstr, 0, 9 * sizeof(char));
  strncpy(numstr, record + 60, 6);
  *beta = (float) atof(numstr);
}

  
/* Break a pdb ATOM record into its fields.  The user must provide the
   necessary space to store the atom name, residue name, and segment name.
   Character strings will be null-terminated.  Returns the atom serial number.
*/
int get_pdb_fields(char *record, char *name, char *resname, char *chain,
		char *segname, char *element, char *resid, char *insertion,
		float *x, float *y, float *z, float *occup, float *beta) {
  int i,len, num, base;
  
  num=0;

  /* get serial number */
  if (record[6] >= 'A' && record[6] <= 'Z') {
    /* If there are too many atoms, XPLOR uses 99998, 99999, A0000, A0001, */
    base = ((int)(record[6] - 'A') + 10) * 100000;
    sscanf(record + 6, "%d", &num);
    num += base;
  } else {
    sscanf(record + 6,"%d",&num);
  }

  /* get atom name */
  strncpy(name,record + 12, 4);
  name[4] = '\0';
  while((len = strlen(name)) > 0 && name[len-1] == ' ')
    name[len-1] = '\0';
  while(len > 0 && name[0] == ' ') {
    for(i=0; i < len; i++)  name[i] = name[i+1];
    len--;
  }

  /* get residue name */
  strncpy(resname,record + 17, 4);
  resname[4] = '\0';
  while((len = strlen(resname)) > 0 && resname[len-1] == ' ')
    resname[len-1] = '\0';
  while(len > 0 && resname[0] == ' ') {
    for(i=0; i < len; i++)  resname[i] = resname[i+1];
    len--;
  }

  chain[0] = record[21];
  if ( chain[0] == ' ' ) chain[0] = 0;
  else chain[1] = 0;

  /* get residue id number plus insertion code */
  strncpy(resid,record + 22, 4);
  resid[4] = '\0';
  while((len = strlen(resid)) > 0 && resid[len-1] == ' ')
    resid[len-1] = '\0';
  if ( record[26] != ' ' ) {
    resid[len] = record[26];
    resid[++len] = '\0';
  }
  while(len > 0 && resid[0] == ' ') {
    for(i=0; i < len; i++)  resid[i] = resid[i+1];
    len--;
  }

  insertion[0] = record[26];
  insertion[1] = 0;

  /* get x, y, and z coordinates */
  get_pdb_coordinates(record, x, y, z, occup, beta);

  /* get segment name	*/
  if(strlen(record) >= 73) {
    strncpy(segname, record + 72, 4);
    segname[4] = '\0';
    while((len = strlen(segname)) > 0 && segname[len-1] == ' ')
      segname[len-1] = '\0';
    while(len > 0 && segname[0] == ' ') {
      for(i=0; i < len; i++)  segname[i] = segname[i+1];
      len--;
    }
  } else {
    strcpy(segname,"");
  }
   
  /* get element name	*/
  if(strlen(record) >= 77) {
    strncpy(element, record + 76, 2);
    element[2] = '\0';
    while((len = strlen(element)) > 0 && element[len-1] == ' ')
      element[len-1] = '\0';
    while(len > 0 && element[0] == ' ') {
      for(i=0; i < len; i++)  element[i] = element[i+1];
      len--;
    }
  } else {
    strcpy(element,"");
  }
   
  return num;
}  


void write_pdb_remark(FILE *outfile, const char *comment) {

  fprintf(outfile,"REMARK %s\n",comment);

}

void write_pdb_end(FILE *outfile) {

  fprintf(outfile,"END\n");

}

void write_pdb_atom(FILE *outfile,
    int index,char *atomname,char *resname,int resid, char *insertion, float x,
    float y, float z, float occ, float beta, char *chain, char *segname,
    char *element) {

  char name[6], rname[5], sname[5];
  char chainc, insertionc;
  int p;

  name[0] = ' ';
  strncpy(name+1, atomname, 4);
  name[5] = '\0';
  if ( strlen(name) == 5 ) {
    atomname = name + 1;
  } else {
    atomname = name;
    while((p = strlen(name)) < 4) {
      name[p] = ' ';
      name[p+1] = '\0';
    }
  }

  strncpy(rname, resname, 4);
  rname[4] = '\0';
  resname = rname;

  chainc = ( chain[0] ? chain[0] : ' ' );
  resid = resid % 10000;
  insertionc = ( insertion[0] ? insertion[0] : ' ' );

  strncpy(sname, segname, 4);
  sname[4] = '\0';
  segname = sname;

  if (index < 100000) {
    fprintf(outfile,
      "%s%5d %4s%c%-4s%c%4d%c   %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%2s\n",
          "ATOM  ", index, atomname, ' ', resname, chainc, resid,
          insertionc, x, y, z, occ, beta, segname, element);
  } else {
    fprintf(outfile,
      "%s***** %4s%c%-4s%c%4d%c   %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%2s\n",
          "ATOM  ", atomname, ' ', resname, chainc, resid,
          insertionc, x, y, z, occ, beta, segname, element);
  }
}


