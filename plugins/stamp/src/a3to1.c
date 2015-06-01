/******************************************************************************
 The computer software and associated documentation called STAMP hereinafter
 referred to as the WORK which is more particularly identified and described in 
 the LICENSE.  Conditions and restrictions for use of
 this package are also in the LICENSE.

 The WORK is only available to licensed institutions.

 The WORK was developed by: 
	Robert B. Russell and Geoffrey J. Barton

 Of current addresses:

 Robert B. Russell (RBR)             Geoffrey J. Barton (GJB)
 Bioinformatics                      EMBL-European Bioinformatics Institute
 SmithKline Beecham Pharmaceuticals  Wellcome Trust Genome Campus
 New Frontiers Science Park (North)  Hinxton, Cambridge, CB10 1SD U.K.
 Harlow, Essex, CM19 5AW, U.K.       
 Tel: +44 1279 622 884               Tel: +44 1223 494 414
 FAX: +44 1279 622 200               FAX: +44 1223 494 468
 e-mail: russelr1@mh.uk.sbphrd.com   e-mail geoff@ebi.ac.uk
                                     WWW: http://barton.ebi.ac.uk/

 The WORK is Copyright (1997,1998) Robert B. Russell & Geoffrey J. Barton
	
	
	

 All use of the WORK must cite: 
 R.B. Russell and G.J. Barton, "Multiple Protein Sequence Alignment From Tertiary
  Structure Comparison: Assignment of Global and Residue Confidence Levels",
  PROTEINS: Structure, Function, and Genetics, 14:309--323 (1992).
*****************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define acids3 "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL ASX GLX UNK CYH ADE CYT GUA THY URA A    A    A C    C    C G    G    G T    T    T U    U    U 1MA MAD T6A OMC 5MC OMG 1MG 2MG M2G 7MG G7M 5MU RT   RT 4SU DHU H2U PSU I    I    I YG   YG QUO"
#define acids1 "ARNDCQEGHILKMFPSTWYVBZXcACGTUAAACCCGGGTTTUUUEEFBHJKLRMMTTTSDDPIIIYYQ"

/* Converts three letter amino acid code to one letter
 *  amino acid code 
 * Note: CYS refers to cystine, CYH refers to cysteine */

char a3to1(char *a3) {
	int i;
	char new;

	new='X';
	for(i=0;i<68; i++) {
	   if (strncmp(&acids3[i*4],a3,3) == 0) 
	      new=(char)acids1[i];
	} 

	printf("%s: %c\n",a3,new);

	return(new);
} 
