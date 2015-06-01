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

/* Calculates the probability matrix, prob, after Rossmann and Argos, given
 * the two sets of coordinates, (float)atoms1 and (float)atoms2. 
 *
 * Modification January 1995, Old shite version was inefficient.  Now
 *  fewer redundent boolean tests are done, and most importantly, the
 *  matrix is only navigated twice if absolutely necessary */

int probcalc(int **atoms1, int **atoms2, int **prob, int lena, int lenb,
	     struct parameters *parms) {
  
  int i,j/*,k*/,ii,jj;
  float sum,sumsq;
  float Dij,Cij,mean,sd,const1,const2;
/*  int start,end,ll; */
  
  
  const1=parms[0].const1;
  const2=parms[0].const2;
  
  
  
  if(parms[0].BOOLEAN) {
    for(i=0; i<lena; i++) {
      ii=i+1;
      for(j=0; j<lenb; j++)  {
	jj=j+1;
	/* The following calculates a Probability matrix after Rossmann and
	 *  Argos (J.Mol.Biol., 105_, 75 (1976))...
	 * The routine 'rossmann' returns both the probabilty Pij, and the
	 *  pure distance parameter Di */
	prob[ii][jj]=(rossmann(&atoms1[i],&atoms2[j], (i==0 || j==0),(i==lena-1 || j==lenb-1),
			       const1,const2,&Dij,&Cij,parms[0].PRECISION)>=parms[0].BOOLCUT);
      } 
    }  
  } else if(!parms[0].STATS) {
    /* using fixed mean and sd, don't need to calculate mean or standard deviation */
    mean=parms[0].NMEAN;
    sd=parms[0].NSD;
    for(i=0; i<lena; i++) {
      ii=i+1;
      for(j=0; j<lenb; j++)  {
	jj=j+1;
	/* The following calculates a Probability matrix after Rossmann and
	 *  Argos (J.Mol.Biol., 105_, 75 (1976))...
	 * The routine 'rossmann' returns both the probabilty Pij, and the
	 *  pure distance parameter Dij.  */
	prob[ii][jj]=(int)
	  ((float)parms[0].PRECISION*(rossmann(&atoms1[i],&atoms2[j],
					       (i==0 || j==0),(i==lena-1 || j==lenb-1),
					       const1,const2,&Dij,&Cij,parms[0].PRECISION) - mean)/sd);
      }      
    }
  } else {
    sum=sumsq=0.0;
    for(i=0; i<lena; i++) {
      ii=i+1;
      for(j=0; j<lenb; j++)  {
	jj=j+1;
	/* The following calculates a Probability matrix after Rossmann and
	 *  Argos (J.Mol.Biol., 105_, 75 (1976))...
	 * The routine 'rossmann' returns both the probabilty Pij, and the
	 *  pure distance parameter Dij.  */
	prob[ii][jj]=(int)((float)parms[0].PRECISION*rossmann(&atoms1[i],&atoms2[j],
							      (i==0 || j==0),(i==lena-1 || j==lenb-1),
							      const1,const2,&Dij,&Cij,parms[0].PRECISION));
	
	sum+=(float)prob[ii][jj]; 
	sumsq+=(float)(prob[ii][jj]*prob[ii][jj]);
      }
    }  
    mean=((float)parms[0].PRECISION*(sum/(float)(lena*lenb)));
    sd=(float)parms[0].PRECISION*
      (float)sqrt( (sumsq-(sum*sum)/(float)(lena*lenb)) / (lena*lenb-1) );
    /* Now we must find out how many SD's above the mean each value
     *  in the probability matrix is. */
    for(i=0; i<lena; i++) {
      ii=i+1;
      for(j=0; j<lenb; ++j) {
	jj=j+1;
	prob[ii][jj]=(int)( (float)parms[0].PRECISION*((float)(prob[ii][jj]-mean)/(float)(sd)));
      }
    }
  }
  
  return 0;
  
} 
