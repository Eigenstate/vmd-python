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
#include <stamp.h>

int comp();

/*************************************************************************
dosort:  Version 2:  create the sortarr array, then copy the values
of each result into the array, freeing the memory as we go.
do qsort on the sortarr array, and return a pointer to the head of the 
array.
This  is a revised version that creates the sortarr array as we destroy the
result array.
-------------------------------------------------------------------------*/
/*****************************************************************************
  Also developed by: John Eargle
  Copyright (2004) John Eargle
*****************************************************************************/
struct path *dosort(struct olist *result, int *lena, int *total) {

  int i,j;
  int k=0;
  struct path *sortarr;
  
  /* JE rm{ */
  sortarr = (struct path *) malloc(sizeof(struct path));
  /* } */
  
  /* JE { */
  sortarr = (struct path *) malloc(*total * sizeof(struct path));
  /* } */
  
  for(i=0; i < ((*lena)-1); ++i){
    if(result[i].len > 0){	    /* if there are paths in this row */
      for(j=0; j < result[i].len; ++j){
	
	/* JE rm{
	sortarr = (struct path *) 
	frealloc(sortarr,sizeof(struct path) *(k+1));
	}
	*/
	
	sortarr[k++] = result[i].res[j];  
      }
    }
    /*	free(result[i].res);  */
  }
  
  /*    free(result); */
  
  /*    if(k != *total) printf("k != total in dosort"); */
  
  if(*total > 0){
    qsort((char *) sortarr, k, sizeof(struct path), comp);
  }
  return sortarr;
  
}

/*************************************************************************
comp:  compare two scores in the sortarr array
-------------------------------------------------------------------------*/
int comp(left,right)
     struct path *left, *right;
{
  return right->score - left->score;
}
