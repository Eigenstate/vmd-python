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
#include <math.h>

/* special integer atoms version for STAMP */
int matmult(float **r, float *v, int **coord, int n, int PRECISION) {

      int i,j,k;
      float temp;
      float t[3];
	
/*	printmat(r,v,3,stdout); */

      for(k=0; k<n; ++k) {
	for(i=0; i<3; ++i) 
	   t[i]=(float)coord[k][i]/(float)PRECISION;
/*	for(i=0; i<3; ++i)
	   printf("%8d ",coord[k][i]);
	printf("--> "); */

	for(i=0; i<3; ++i) {
	   temp=0.0;
	   for(j=0; j<3; ++j) 
	      temp+=r[i][j]*t[j]; 
	   coord[k][i]=(int)( (float)PRECISION*((float)temp+v[i]));
	}  /* End of for(i=... */
/*	for(i=0; i<3; ++i)
           printf("%8d ",coord[k][i]);
	printf("\n"); */

      }  /* End of for(k.... */

      return 0;

} /* End of function */
	       
