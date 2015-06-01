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
#include <stamp.h>

/* REVerse of MATMULT
 *
 * Since with matmult we assume the following is true:
 *
 *     X2 = R*X1 + V
 *
 *  to get X1 back we must do:
 *
 *     X1 = R^-1*(X2 - V)
 *
 *  where R^-1 is the inverse of R 
 *
 * That is what this routine does.  It makes use of a routine called MATINV from
 *  Numerical recipes.
 *
 * "R" and "V" above are the "r" and "v" supplied below.
 * X2 and X1 are both *integer* coordinate sets.
 *
 * RBR November 1992
 */
int revmatmult(float **r, float *v, int **coord, int n, int PRECISION) {

	int i,j,k;
	int *indx;

	float temp;
	float t[3];
	float **R,**RI;
	float pos;


	pos=1.0;
	/* getting the inverse matrix */
	indx=(int*)malloc(4*sizeof(int));
	RI=(float**)malloc(4*sizeof(float*));
	R=(float**)malloc(4*sizeof(float*));
	for(i=0; i<4; ++i) {
	   RI[i]=(float*)malloc(4*sizeof(float));
	   R[i]=(float*)malloc(4*sizeof(float));
	}
	for(i=0; i<3; ++i)  {
	  for(j=0; j<3; ++j) {
	    R[i+1][j+1]=r[i][j];
	  }
	}

	matinv(R,RI,&pos,indx);


      for(k=0; k<n; ++k) {
	for(i=0; i<3; ++i) 
	   t[i]=((float)coord[k][i]/(float)PRECISION) - v[i];

	for(i=0; i<3; ++i) {
	   temp=0.0;
	   for(j=0; j<3; ++j) 
	      temp+=RI[i+1][j+1]*t[j]; 
	   coord[k][i]=(int)( (float)PRECISION*((float)temp));
	}  /* End of for(i=... */

      }  /* End of for(k.... */

      for(i=0; i<4; ++i) {
	free(RI[i]);
	free(R[i]);
      }
      free(R); free(RI); free(indx);
      return 0;

} /* End of function */
	       
