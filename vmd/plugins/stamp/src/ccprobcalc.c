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

#include <stamp.h>

int ccprobcalc(int **atoms1, int **atoms2, int **prob, int lena, int lenb, 
	struct parameters *parms)

/* Calculates the probability matrix, prob, after Rossmann and Argos, given
 * the two sets of coordinates, (float)atoms1 and (float)atoms2. 
 *
 * Modified version of probcalc to corner cut.  This avoids calculating
 *  part of the matrix to save time */
{

        int i,j,k,ii,jj;
	int start,end,ll;
	int upper,uval,lstrt,lstart;

	float sum,sumsq;
        float Dij,Cij,mean,sd,const1,const2;
	float fcut,fm,fn,jt,fk1;

	const1=parms[0].const1;
	const2=parms[0].const2;

	mean=((float)parms[0].PRECISION*parms[0].NMEAN); 
	sd=((float)parms[0].PRECISION*parms[0].NSD); 
        sum=sumsq=0.0;
	

	if(parms[0].CCADD) {
          /* auto corner cutting defined so derive fk1 from lengths */
	  fk1 = parms[0].CCFACTOR + fabs((float)(lena-lenb-4));
	} else {
	  fk1 = parms[0].CCFACTOR;
	}
	fn = (float) lena+1;  /* len + 1 */
	fm = (float) lenb+1;
	fcut = (fk1/sqrt(2.0))*(((fm/fn)+(fn/fm))/sqrt(fm*fm+fn*fn));
	upper = 0;
/*	fprintf(parms[0].LOG,"CC factor=%f, |lena-lenb-4|=%f\n",parms[0].CCFACTOR,fabs((float)(lena-lenb-4)));
	fprintf(parms[0].LOG,"CC variables: m=%f, n=%f, fk1=%f, distance=%f\n",fm,fn,fk1,fcut);
*/

	
	lstart=0;
	for(j=0; j<lena; ++j) {
	   /* corner cutting */
	   jt=j/fn; 
	   jj=j+1;
           for(i=lstart; i<lenb; ++i)  {
	     ii=i+1;
	     if(fabs((i/fm)-jt) > fcut) {
	       if(upper) {
		 goto l1;
	       } else {
		 goto l2;
	       }
	    } else {
	      if(!upper) {
		lstrt = i;
		upper = 1;
	      }
	    }
	    if(!parms[0].BOOLEAN) {
               prob[jj][ii]=(int)((float)parms[0].PRECISION*rossmann(&atoms1[j],&atoms2[i],
			   (i==0 || j==0),(j==lena-1 || i==lenb-1),
			   const1,const2,&Dij,&Cij,parms[0].PRECISION)); 
               prob[jj][ii]=(int)( (float)parms[0].PRECISION*((float)((float)prob[jj][ii]-mean)/(float)(sd)));
	       sum+=(float)prob[jj][ii]; sumsq+=(float)(prob[jj][ii]*prob[jj][ii]); 
/*		prob[jj][ii]=10*parms[0].PRECISION; */
	    } else {
	       prob[jj][ii]=(rossmann(&atoms1[j],&atoms2[i], (i==0 || j==0),(j==lena-1 || i==lenb-1),
			       const1,const2,&Dij,&Cij,parms[0].PRECISION)>=parms[0].BOOLCUT);
	    }
	    l2:; /* jump to here if corner cutting */
         }  /* for (i=... */
	 l1: /* jump to here if corner cutting */

	 upper = 0;
	 lstart=lstrt;
     	}  /* for (j=0... */

	return 0;

} /* End of function ccprobcalc */
