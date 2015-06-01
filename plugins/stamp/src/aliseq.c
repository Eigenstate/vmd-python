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
#include <stamp.h>


/*****************************************************************************
 * aliseq:  given the patha array, seqa, seqb, and path details, return an
 *          alignment of the sub-regions of seqa and seqb indicated by the path.
 *          works with strings of '1's and ' 's
 *  Arguments:
 *    char *seqa -> char* for the first sequence
 *    char *seqb -> char* for the second sequence
 *    struct path *apath -> 
 *    unsigned char **patha -> 
 *    char *alseq <- 
 *    char *blseq <- 
 *    int *k <- 
 *    int *hgap <- 
 *    int *vgap <- 
 *  Return:
 *    int 0 for success or failure (weird)
-----------------------------------------------------------------------------*/
int aliseq(char *seqa, char *seqb, struct path *apath, unsigned char **patha, 
	   char *alseq, char *blseq, int *k, int *hgap, int *vgap) {
  
  unsigned char DIAG = 01; /* mask for diagonal move */
  unsigned char HORIZ= 02; /*          horizontal    */
  unsigned char VERT = 04; /*          vertical      */
  unsigned char GAP  = ' ';
  
/*  int ii,jj; */
  int i = apath->end.i;
  int j = apath->end.j;
/*  char temp; */
  /*
    for(ii=0; ii<apath->end.i; ++ii) {
    for(jj=0; jj<apath->end.j; ++jj) 
    printf("%1d",(int)patha[jj][ii]);
    printf("\n");
    }
  */
  
  *k = -1; *hgap = 0; *vgap = 0;
  
  /* JE {
     Make sure the last element in the aligned path is added
  */
  /*
    if ( i > apath->start.i-1 || j > apath->start.j-1 ) {
    ++*k;
    alseq[*k] = seqa[i-1];
    blseq[*k] = seqb[j-1];
    i--;
    j--;
    }
  */
  /*
   }
  */
  
  while ( i > apath->start.i-1 || j > apath->start.j-1 ) {
    if((DIAG & patha[j][i]) == DIAG) {
      printf("[%d,%d] ",i,j);
      ++*k;
      alseq[*k] = seqa[i-1];
      blseq[*k] = seqb[j-1];
      i--; j--;
    } else if((HORIZ & patha[j][i]) == HORIZ) {
      ++*k;
      alseq[*k] = GAP;
      *hgap += 1;
      blseq[*k] = seqb[j-1];
      j--;
    } else if((VERT & patha[j][i]) == VERT) {
      ++*k;
      alseq[*k] = seqa[i-1];
      blseq[*k] = GAP;
      *vgap += 1;
      i--;
    } else {
      printf("Disaster in aliseq\n");
      return 0;
    }
  }
  
  /* JE rm{
     Implicit DIAG at the start of the path
  */
  /*
    ++*k;
    alseq[*k] = seqa[i-1];
    blseq[*k] = seqb[j-1];
  */
  /* } */
  
  reval(alseq,0,*k);
  reval(blseq,0,*k);
  ++(*k);
  alseq[*k] = '\0';
  blseq[*k] = '\0';
  return 0;
}

