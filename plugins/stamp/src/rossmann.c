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
/*****************************************************************************
  Also developed by: John Eargle
  Copyright (2004) John Eargle
*****************************************************************************/
#include <stdio.h>
#include <math.h>
#include <stamp.h>

/* This function calculates the Rossmann and Argos probability for a 
 *  single pair of (integer) coordinates.  Pij is returned, as
 *  are Dij and Cij (the distance and conformational components of
 *  Pij respectively */

float rossmann(int **atoms1, int **atoms2, int start, int end,
	       float const1, float const2, float *Dij, float *Cij, int PRECISION) {
  
/*  int i,j; */
  float dijsq,Sijsq,t1,t2,t3,t4,t5,t6;
  float dx,dy,dz;
  float Pij;
  
  if(start) t1=t2=t3=0;
  else {

    /* JE { */
    t1=(float)((atoms1[-1][0] - atoms1[0][0]) - (atoms2[-1][0] - atoms2[0][0]));
    t2=(float)((atoms1[-1][1] - atoms1[0][1]) - (atoms2[-1][1] - atoms2[0][1]));
    t3=(float)((atoms1[-1][2] - atoms1[0][2]) - (atoms2[-1][2] - atoms2[0][2]));
    /* } */

  } /* End of if(start)... */
  t1*=t1; t2*=t2; t3*=t3;
  
  if(end) t4=t5=t6=0;
  else {

    /* JE { */
    t4=(float)((atoms1[1][0] - atoms1[0][0]) - (atoms2[1][0] - atoms2[0][0]));
    t5=(float)((atoms1[1][1] - atoms1[0][1]) - (atoms2[1][1] - atoms2[0][1]));
    t6=(float)((atoms1[1][2] - atoms1[0][2]) - (atoms2[1][2] - atoms2[0][2]));
    /* } */

    t4*=t4; t5*=t5; t6*=t6;
  } /* End of if(end).... */
  
  dx=(float)(atoms1[0][0] - atoms2[0][0]);
  dy=(float)(atoms1[0][1] - atoms2[0][1]);
  dz=(float)(atoms1[0][2] - atoms2[0][2]);
  
  dijsq=dx*dx+dy*dy+dz*dz;
  Sijsq=t1+t2+t3+t4+t5+t6;
  
  /* conversion to floating point occurs at this stage */
  (*Dij)=(dijsq/(float)(PRECISION*PRECISION))/const1;
  (*Cij)=(Sijsq/(float)(PRECISION*PRECISION))/const2;

  /* JE rm{ */
  Pij=(float)exp((double)((*Dij)+(*Cij)));
  /* } */

  /* JE {
  Pij = (float)(exp((double)*Dij) * exp((double)*Cij));
   }
  */

  /*	    printf("Pij=%f\n",Pij); */
  return Pij;
}
