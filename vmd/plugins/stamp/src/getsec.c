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
#include <stamp.h>

/* Get secondary structures from a file */

int getsec(struct domain_loc *domain, int ndomain, struct parameters *parms) {

	char c;
	char id[100];
	char *sec;
	char *empty=0;

	int i,j/*,k*/;
	int which;

	int *found;

	FILE *IN;

	if((IN=fopen(parms[0].secfile,"r"))==NULL) {
	   fprintf(stderr,"error: file %s does not exist\n",parms[0].secfile);
	   for(i=0; i<ndomain; ++i) {
	     for(j=0; j<domain[i].ncoords; ++j) domain[i].sec[j]='?';
	     domain[i].sec[j]='\0';
	   }
	   return -1;
	}
	fprintf(parms[0].LOG,"Searching for secondary structure assignments in the file %s\n",
		parms[0].secfile);
	sec=(char*)malloc(parms[0].MAX_SEQ_LEN*sizeof(char));
	found=(int*)malloc(ndomain*sizeof(int));

	for(i=0; i<ndomain; ++i) found[i]=0;
	
	while((c=getc(IN))!=(char)EOF) {
	   if(c=='>') {  
	      if(fscanf(IN,"%s",&id[0])==(int)EOF) return -1; 
	      which=-1;
	      for(i=0; i<ndomain; ++i) 
		 if(strcmp(id,domain[i].id)==0 && !found[i]) { which=i; found[i]=1; break; }
	      while((c=getc(IN))!=(char)EOF && c!='\n') if(c==(char)EOF) break;
	      while((c=getc(IN))!=(char)EOF && c!='\n') if(c==(char)EOF) break;
	      i=0;
	      while((c=getc(IN))!=(char)EOF && c!='*') { 
		if(c==(char)EOF) break;
		if(c!='\n') sec[i++]=c;
		if(i>parms[0].MAX_SEQ_LEN) break;
	      }
	      sec[i]='\0';
	      if(which!=-1) strcpy(domain[which].sec,sec);
           }
	}
	/* now see how many we missed */
	for(i=0; i<ndomain; ++i) {
	   if(found[i]) {
	      fprintf(parms[0].LOG,"\nFound secondary structure for %s\n",domain[i].id);
	      if(strlen(domain[i].sec)!=strlen(domain[i].aa)) {
		fprintf(parms[0].LOG," though the lengths differ, ");
		if(strlen(domain[i].sec)>strlen(domain[i].aa)) {
		   domain[i].sec[strlen(domain[i].aa)]='\0';
		   fprintf(parms[0].LOG,"truncating\n");
		} else {
		  for(j=strlen(domain[i].sec); j<strlen(domain[i].aa); ++j) domain[i].sec[j]='?';
		  domain[i].sec[strlen(domain[i].aa)]='\0';
		  fprintf(parms[0].LOG,"padding with ?s\n");
		}
		fprintf(parms[0].LOG," warning: assignment of secondary structure to sequence may be incorrect\n");
	     }
	   } else {
	      fprintf(parms[0].LOG,"no secondary structure found for %s\n",domain[i].id);
	      for(j=0; j<strlen(domain[i].aa); ++j) domain[i].sec[j]='?';
	   }
	   display_align(&domain[i].aa,1,&domain[i].sec,1,&domain[i].aa,&domain[i].sec,
	       empty,empty,parms[0].COLUMNS,0,0,parms[0].LOG);

	}
	return 0;
}
