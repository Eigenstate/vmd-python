
/***************************************************************************
 * DESCRIPTION:
 *
 * General routines to read .pdb files.
 *
 ***************************************************************************/

#ifndef READ_PDB_H
#define READ_PDB_H

#include <stdio.h>

#define PDB_RECORD_LENGTH	80

/*	record type defines	*/
enum {PDB_REMARK, PDB_ATOM, PDB_UNKNOWN, PDB_END, PDB_EOF, PDB_CRYST1};

/* read the next record from the specified pdb file, and put the string found
   in the given string pointer (the caller must provide adequate (81 chars)
   buffer space); return the type of record found
*/
int read_pdb_record(FILE *f, char *retStr);

/* get the CRYST1 information about the unit cell (but not space group) */
void get_pdb_cryst1(char *record, float *alpha, float *beta, float *gamma,
		    float *a, float *b, float *c);

/* Extract the x,y, and z coordinates from the given ATOM record.	*/
void get_pdb_coordinates(char *record, float *x, float *y, float *z,
	float *occup, float *beta);

/* Break a pdb atom record into it's fields.  The user must provide the
   necessary space to store the atom name, residue name, and segment name.
   Character strings will be null-terminated.  Returns the atom serial number. */
int get_pdb_fields(char *record, char *name, char *resname, char *chain, 
		char *segname, char *element, char *resid, char *insertion,
		float *x, float *y, float *z, float *occup, float *beta);

/* Write a remark to a pdb file. */

void write_pdb_remark(FILE *outfile, const char *comment);

/* Write end to a pdb file */

void write_pdb_end(FILE *outfile);

/* Write a pdb file atom record */

void write_pdb_atom(FILE *outfile,
    int index,char *atomname,char *resname,int resid, char *insertion, float x,
    float y, float z, float occ, float beta, char *chain, char *segname,
    char *element);

#endif

