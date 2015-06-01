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

/* This checks for compression, etc and runs open/popen as appropriate */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

FILE *openfile(char *filename, char *type) {

	FILE *handle;
	char command[1000];

	if(strcmp(type,"r")==0) { /* A read file */
		if(strcmp(&filename[strlen(filename)-2],".Z")==0) { /* UNIX compression */
#if defined(_MSC_VER)
                     return NULL;
#else 
/*		     sprintf(command,"zcat %s 2> /dev/null",filename);  */
		     sprintf(command,"zcat %s",filename);  
		     handle=popen(command,"r");
#endif
		} else if(strcmp(&filename[strlen(filename)-3],".gz")==0) { /* gzipped  */
/*		     sprintf(command,"gunzip -c %s 2> /dev/null",filename);  */
		     sprintf(command,"gunzip -c %s",filename); 
#if defined(_MSC_VER)
                     return NULL;
#else 
		     handle=popen(command,"r");
#endif
		} else { /* presume no compression */
		    handle=fopen(filename,"r");
		}
	} else {
	  	    handle=fopen(filename,"w");
	}
	return handle;
}

	
