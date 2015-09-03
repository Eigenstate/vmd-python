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
/**********************************************************
      MATFIT

      Note that XA is ATOMS1 and XB is ATOMS2   G.J.B.

      SUBROUTINE TO FIT THE COORD SET ATOMS1(3,N) TO THE SET ATOMS2(3,N)
      IN THE SENSE OF:
             XA= R*XB +V
      R IS A UNITARY 3.3 RIGHT HANDED ROTATION MATRIX
      AND V IS THE OFFSET VECTOR. THIS IS AN EXACT SOLUTION

     THIS SUBROUTINE IS A COMBINATION OF MCLACHLAN'S AND KABSCH'S
     TECHNIQUES. SEE
      KABSCH, W. ACTA CRYST A34, 827,1978
      MCLACHAN, A.D., J. MOL. BIO. NNN, NNNN 1978
      WRITTEN BY S.J. REMINGTON 11/78.

      THIS SUBROUTINE USES THE IBM SSP EIGENVALUE ROUTINE 'EIGEN'

     N.B. CHANGED BY R.B.R (OCTOBER 1990) FOR USE IN THE PROGRAM
      STRUCALIGN.  SEE STRUCALIGN.F FOR DETAILS

     CHANGED BY R.B.R. (JANUARY 1991) FOR USE IN THE PROGRAM
        RIGIDBODY.
 -----------------------------------------------------------
     Translated into C by RBR in January 1991
*************************************************************/
#include <stdio.h>
#include <math.h>

/*  This function mimics the first part of the FORTRAN routine
 *    'matfit.f'.  It makes use of the FORTRAN routine 'qkfit.f'.
 *     This keeps complications involving passing great heaps
 *     of memory to FORTRAN from arising.  The resulting transformation
 *     is in the sense of:
 *	   atoms1 = R*atoms2 + V     
 *
 *
 * Special version for integer atoms */

/* float idist(); */

/* SMJS Added */
#include <stamp.h>

float matfit(int **atoms1, int **atoms2, float **R, float *V,
	int nats, int entry, int PRECISION) {

	/* Note that everything that is passed to the routine
	 *  qkfit must be double precision. */

	/*  double *cma, *cmb, **umat;
	    double **r; */

	double cma[3], cmb[3];
	double umat[3][3];
	double r[3][3]; 
	double xasq, xbsq, xni,t,rtsum,rmse;
/*	float distance; */
	int i,j,k,xn;

/* SMJS Added */
        long long_entry;


/*
	cma=(double*)malloc(3*sizeof(double));
	cmb=(double*)malloc(3*sizeof(double));
	r=(double**)malloc(3*sizeof(double*));
	umat=(double**)malloc(3*sizeof(double*));
	for(i=0; i<3; ++i) {
		r[i] = (double*)malloc(3*sizeof(double));
		umat[i] = (double*)malloc(3*sizeof(double));
	}
*/
	xn=nats;
	xasq=0.0;
	xbsq=0.0;
	xni=1.0/((double)xn);



/* Accumulate uncorrected (for c.m.) sums and squares */

	for(i=0; i<3; ++i) {
	   cma[i]=cmb[i]=0.0;
	   for(j=0; j<=2; ++j) umat[j][i]=0.0;

           for(j=0; j<nats; ++j) {
	      for(k=0; k<3; ++k) {
	        umat[k][i]+=(double)atoms1[j][i]/(double)PRECISION*(double)atoms2[j][k]/(double)PRECISION;
	      }  
/*
	      printf("matfit fitting: ");
	      for(k=0; k<3; ++k) printf("%10d ",atoms1[j][k]);
	      printf("and ");
	      for(k=0; k<3; ++k) printf("%10d ",atoms2[j][k]); 
	      distance = idist(atoms1[j],atoms2[j],PRECISION);
	      printf("dist %f \n",distance);
*/

	      
               
		
	      t=(double)atoms1[j][i]/(double)PRECISION;
	      cma[i]+=t;
	      xasq+=t*t;
	      t=(double)atoms2[j][i]/(double)PRECISION;
	      cmb[i]+=t;
	      xbsq+=t*t;

	   }
	}

/* Subtract cm offsets */

	for(i=0; i<=2; ++i) {
	   xasq-=cma[i]*cma[i]*xni;
	   xbsq-=cmb[i]*cmb[i]*xni;
	   for(j=0; j<=2; ++j) {
	       umat[j][i]=(umat[j][i]-cma[i]*cmb[j]*xni)*xni;
	       r[i][j] = 0.0;
	   }
	} 

/* Fit it */

/* SMJS changed entry to long_entry */
        long_entry = entry;
	qkfit(&umat[0][0],&rtsum,&r[0][0],&long_entry); 
/* SMJS Added */
        entry = (int) long_entry;

/*	qkfit(**umat,&rtsum,**r,&entry);  */
	rmse=(xasq+xbsq)*xni - 2.0*rtsum;
	if(rmse<0.0) rmse=0.0;
	rmse=sqrt(rmse);

/* The matrix obtained from the FORTRAN routine (r) must be transfered to
 *  the 'malloc'ed C matrix (R).  */
	for(i=0; i<3; ++i) {
	   for(j=0; j<3; ++j)  {
/*		printf("R[%d][%d] = %10f => %10f \n",i,j,R[i][j],r[j][i]);   */
		R[i][j] = (float) r[j][i];
/*		R[i][j] = r[j][i];  */
	   }
/*	   printf("\n");  */
	}

/* printf("IN MATFIT RMSE = %f\n",rmse);  */


/* Calculate offset if entry=1 */


	if(!entry) return rmse; 

	for(i=0; i<=2; ++i) {
	   t=0.0;
	   for(j=0; j<=2; ++j)  t+=((double)R[i][j])*cmb[j];
	   V[i]=(float)(cma[i]-t)*xni;
	}
/*
	for(i=0; i<3; ++i)  {
		free(r[i]);
		free(umat[i]);
	}
	free(r);
	free(cmb);
	free(cma);
*/

	return rmse;

} 
 
