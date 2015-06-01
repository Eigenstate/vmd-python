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
#include <stdlib.h>
#include <math.h>
#define TINY 1.0e-20

/* Inverts a 3 x 3 matrix. This routine was written with reference to
 *   reading Numerical Recipies in C. 
 * W.H Press, B.P Flannery, S.A. Teukolsky and W.T. Vetterling, 
 * "Numerical Recipes in C: The Art of Scientific Computing",    
 * Cambridge University Press, 1988. */

void lubksb(float **A, int n, int *indx, float b[]);
int ludcmp(float **a, int n, int *indx, float *d);
float *vector(int nl, int nh);
void free_vector(float *v, int nl, int nh);


void matinv(float **a, float **y, float *d, int *indx) {

int i,j,N;
float *col;



col=(float*)malloc(10*sizeof(float));
N=3;

if(ludcmp(a,N,indx,d)==-1) { /* matrix is singular, so just copy a to y (ie. I^-1 = I) */
   for(j=1; j<=N; j++) { 
      for(i=1; i<=N; i++) {
		y[i][j]=a[i][j];
      }
   }
} else {
   for(j=1; j<=N; j++) {
      for(i=1; i<=N; i++) col[i]=0.0;
      col[j]=1.0;
      lubksb(a,N,indx,col);
      for(i=1; i<=N; i++) y[i][j]=col[i];
   }
}

free(col);


}

void lubksb(float **a, int n, int *indx, float b[])
{
	int i,ii=0,ip,j;
	float sum;

	for (i=1;i<=n;i++) {
		ip=indx[i];
		sum=b[ip];
		b[ip]=b[i];
		if (ii)
		for (j=ii;j<=i-1;j++) sum -= a[i][j]*b[j];
		else if (sum) ii=i;
		b[i]=sum;
	}
	for (i=n;i>=1;i--) {
		sum=b[i];
		for (j=i+1;j<=n;j++) sum -= a[i][j]*b[j];
		b[i]=sum/a[i][i];

	}
}



int ludcmp(float **a, int n, int *indx, float *d)
{
	int i,imax=0,j,k;
	float big,dum,sum,temp;
	float *vv,*vector();
	void free_vector();

	vv=vector(1,n);
	*d=1.0;
	for (i=1;i<=n;i++) {
		big=0.0;
		for (j=1;j<=n;j++)
			if ((temp=fabs(a[i][j])) > big) {
			   big=temp;
			}
		if (big == 0.0)  return -1; /* matrix is singular */
		vv[i]=1.0/big;
	}
	for (j=1;j<=n;j++) {
		for (i=1;i<j;i++) {
			sum=a[i][j];
			for (k=1;k<i;k++) sum -= a[i][k]*a[k][j];
			a[i][j]=sum;
		}
		big=0.0;
		for (i=j;i<=n;i++) {
			sum=a[i][j];
			for (k=1;k<j;k++)
				sum -= a[i][k]*a[k][j];
			a[i][j]=sum;
			if ( (dum=vv[i]*fabs(sum)) >= big) {
				big=dum;
				imax=i;
			}
		}
		if (j != imax) {
			for (k=1;k<=n;k++) {
				dum=a[imax][k];
				a[imax][k]=a[j][k];
				a[j][k]=dum;
			}
			*d = -(*d);
			vv[imax]=vv[j];
		}
		indx[j]=imax;
		if (a[j][j] == 0.0) a[j][j]=TINY;
		if (j != n) {
			dum=1.0/(a[j][j]);
			for (i=j+1;i<=n;i++) {
			   a[i][j] *= dum;
		  	}
		}
	}
	free_vector(vv,1,n);
        return 0;
}
float *vector(int nl, int nh)
{
        float *v;

        v=(float *)malloc((unsigned) (nh-nl+1)*sizeof(float));
        if(!v) {
	    printf("allocation failure in vector()");
	    exit(-1);
	}
	
        return v-nl;
}


void free_vector(float *v, int nl, int nh)
{
        free((char*) (v+nl));
}
