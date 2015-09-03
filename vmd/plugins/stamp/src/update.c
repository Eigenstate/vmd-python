/******************************************************************************
 The computer software and associated documentation called STAMP hereinafter
 referred to as the WORK which is more particularly identified and described in 
 the LICENSE.  Conditions and restrictions for use of
 this package are also in the LICENSE.

 The WORK is only available to licensed institutions.

 The WORK was developed by: 
	Robert B. Russell and Geoffrey J. Barton

 Of current contact addresses:

 Robert B. Russell (RBR)             Geoffrey J. Barton (GJB)
 Bioinformatics                      EMBL-European Bioinformatics Institute
 SmithKline Beecham Pharmaceuticals  Wellcome Trust Genome Campus
 New Frontiers Science Park (North)  Hinxton, Cambridge, CB10 1SD U.K.
 Harlow, Essex, CM19 5AW, U.K.       
 Tel: +44 1279 622 884               Tel: +44 1223 494 414
 FAX: +44 1279 622 200               FAX: +44 1223 494 468
 e-mail: russelr1@mh.uk.sbphrd.com   e-mail geoff@ebi.ac.uk
                                     WWW: http://barton.ebi.ac.uk/

   The WORK is Copyright (1997,1998,1999) Robert B. Russell & Geoffrey J. Barton
	
	
	

 All use of the WORK must cite: 
 R.B. Russell and G.J. Barton, "Multiple Protein Sequence Alignment From Tertiary
  Structure Comparison: Assignment of Global and Residue Confidence Levels",
  PROTEINS: Structure, Function, and Genetics, 14:309--323 (1992).
*****************************************************************************/
#include <stdio.h>

/* This function takes in a change matrix and vector (dR, dV) and 
 *   applies it to an 'old' matrix and vector (R,V).  The result
 *   is copied into the 'old' R and V.  */

void update(float **dR, float **R, float *dV, float *V) {

	int i,j,k;
	float rtemp[3][3],vtemp[3];
/*
	printf("dR,dV:\n");
	printmat(dR,dV,3,stdout);
	printf("R,V:\n");
	printmat(R,V,3,stdout);
*/
	/* Matrix multiplication and vector addition */
	for(i=0; i<3; ++i) {
	   vtemp[i]=0;
	   for(j=0; j<3; ++j) {
	      rtemp[i][j]=0;
	      vtemp[i]+=dR[i][j]*V[j];
	      for(k=0; k<3; ++k)
		 rtemp[i][j]+=dR[i][k]*R[k][j];
	    } /* End of for(j=... */
	} /* End of for(i=.... */

	/* Transfering rtemp into R */
	for(i=0; i<3; ++i) {
	   V[i]=vtemp[i]+dV[i];
	   for(j=0; j<3; ++j) R[i][j]=rtemp[i][j];
	   } /* End of for(i=... */
/*	printf("After combining:\nR,V:\n");
	printmat(R,V,3,stdout); */
}

