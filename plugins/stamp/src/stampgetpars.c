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
#include <stdlib.h>
#include <stdio.h>
#include <stamp.h>

int getpars(FILE *fp, struct parameters *var) {

    char c;

    char *parm;		/* name of following dimension */
    char *dim;		/* dimension */
/*    static char *env; */

    int i,T_FLAG;

/*    FILE *IN; */


    parm = (char*)malloc(200*sizeof(char));
    dim  = (char*)malloc(200*sizeof(char));

    while(fscanf(fp,"%s%s",parm,dim) != (int)EOF) {
	for(i=0; i<strlen(parm); ++i) parm[i]=ltou(parm[i]); /* change to upper case */
	T_FLAG=(dim[0]=='Y' || dim[0]=='y' || dim[0]=='1' || dim[0]=='T' || dim[0]=='t' || dim[0]=='o' || dim[0]=='O');
	/* enables one to write '1', 'YES', 'Yes', 'yes', 'T_FLAG', 'True' or 'true' to 
	 *  set any boolean variable to one */
	if(strcmp(parm,"PAIRPEN") == 0 || strcmp(parm,"PEN")==0 || strcmp(parm,"SECOND_PAIRPEN")==0)
		sscanf(dim,"%f",&var[0].second_PAIRPEN);
	else if(strcmp(parm,"FIRST_PAIRPEN")==0)
		sscanf(dim,"%f",&var[0].first_PAIRPEN);
	else if(strcmp(parm,"MAXPITER") == 0 || strcmp(parm,"MAXSITER") == 0)
		sscanf(dim,"%d",&var[0].MAXPITER);
	else if(strcmp(parm,"MAXTITER") == 0)
		sscanf(dim,"%d",&var[0].MAXTITER);
	else if(strcmp(parm,"TREEPEN") == 0 || strcmp(parm,"SECOND_TREEPEN")==0)
		sscanf(dim,"%f",&var[0].second_TREEPEN);
	else if(strcmp(parm,"FIRST_TREEPEN")==0)
		sscanf(dim,"%f",&var[0].first_TREEPEN);
	else if(strcmp(parm,"SCORETOL") == 0)
		sscanf(dim,"%f",&var[0].SCORETOL);
	else if(strcmp(parm,"CLUSTMETHOD") == 0)
		sscanf(dim,"%d",&var[0].CLUSTMETHOD);
	else if(strcmp(parm,"E1") == 0 || strcmp(parm,"SECOND_E1")==0) {
		sscanf(dim,"%f",&var[0].second_E1);
	} 
	else if(strcmp(parm,"E2") == 0 || strcmp(parm,"SECOND_E2")==0) {
		sscanf(dim,"%f",&var[0].second_E2);
	} else if(strcmp(parm,"FIRST_E1")==0)
		sscanf(dim,"%f",&var[0].first_E1);
	else if(strcmp(parm,"FIRST_E2")==0)
		sscanf(dim,"%f",&var[0].first_E2);
 	else if(strcmp(parm,"NPASS")==0) {
		sscanf(dim,"%d",&var[0].NPASS);
		if(var[0].NPASS!=1 && var[0].NPASS!=2) {
		   fprintf(stderr,"error: NPASS must be either 1 or 2\n");
		   return -1;
		}
	} else if(strcmp(parm,"CUTOFF") == 0 || strcmp(parm,"SECOND_CUTOFF")==0)
		sscanf(dim,"%f",&var[0].second_CUTOFF);
	else if(strcmp(parm,"FIRST_CUTOFF")==0) 
		sscanf(dim,"%f",&var[0].first_CUTOFF);
	else if(strcmp(parm,"TREEPLOT") == 0)
		var[0].TREEPLOT=T_FLAG;
	else if(strcmp(parm,"PAIRPLOT") == 0)
		var[0].PAIRPLOT=T_FLAG;
	else if(strcmp(parm,"NALIGN") == 0)
		sscanf(dim,"%d",&var[0].NALIGN);
	else if(strcmp(parm,"DISPALL") == 0)
		var[0].DISPALL=T_FLAG;
	else if(strcmp(parm,"HORIZ") ==0)
		var[0].HORIZ=T_FLAG;
	else if(strcmp(parm,"ADD") ==0)
		sscanf(dim,"%f",&var[0].ADD);
	else if(strcmp(parm,"NMEAN") ==0)
		sscanf(dim,"%f",&var[0].NMEAN);
	else if(strcmp(parm,"NSD") ==0)
		sscanf(dim,"%f",&var[0].NSD);
	else if(strcmp(parm,"STATS") ==0)
		var[0].STATS=T_FLAG;
	else if(strcmp(parm,"NA") == 0)
		sscanf(dim,"%f",&var[0].NA);
	else if(strcmp(parm,"NB") == 0)
		sscanf(dim,"%f",&var[0].NB);
	else if(strcmp(parm,"NASD") == 0)
		sscanf(dim,"%f",&var[0].NASD);
	else if(strcmp(parm,"NBSD") == 0)
		sscanf(dim,"%f",&var[0].NBSD);
	else if(strcmp(parm,"PAIRWISE") == 0) 
		var[0].PAIRWISE=T_FLAG;
	else if(strcmp(parm,"OPD") ==0)
		var[0].opd = T_FLAG;
	else if(strcmp(parm,"TREEWISE") == 0)
		var[0].TREEWISE=T_FLAG;
	else if(strcmp(parm,"ORDFILE") == 0)
		strcpy(var[0].ordfile,dim);
	else if(strcmp(parm,"TREEFILE") == 0)
		strcpy(var[0].treefile,dim);
	else if(strcmp(parm,"PLOTFILE") == 0)
		strcpy(var[0].plotfile,dim);
	else if(strcmp(parm,"PREFIX") == 0 || strcmp(parm,"TRANSPREFIX")==0)
		strcpy(var[0].transprefix,dim);
	else if(strcmp(parm,"MATFILE") == 0)
		strcpy(var[0].matfile,dim);
	else if(strcmp(parm,"THRESH") ==0)
		sscanf(dim,"%f",&var[0].THRESH);
	else if(strcmp(parm,"TREEALIGN")==0)
		var[0].TREEALIGN=T_FLAG;
	else if(strcmp(parm,"TREEALLALIGN")==0)
		var[0].TREEALLALIGN=T_FLAG; 
	else if(strcmp(parm,"PAIRALIGN")==0 || strcmp(parm,"SCANALIGN")==0) 
		var[0].PAIRALIGN=T_FLAG;
	else if(strcmp(parm,"PAIRALLALIGN")==0 || strcmp(parm,"SCANALLALIGN")==0)
		var[0].PAIRALLALIGN=T_FLAG;
	else if(strcmp(parm,"PRECISION")==0)
		sscanf(dim,"%d",&var[0].PRECISION);
	else if(strcmp(parm,"MAX_SEQ_LEN")==0)
		sscanf(dim,"%d",&var[0].MAX_SEQ_LEN);
	else if(strcmp(parm,"ROUGHFIT")==0)
		var[0].ROUGHFIT=T_FLAG;
	else if(strcmp(parm,"ROUGHOUT")==0) 
		var[0].roughout=T_FLAG;
	else if(strcmp(parm,"ROUGHOUTFILE")==0) { 
		strcpy(&var[0].roughoutfile[0],dim);
		var[0].roughout=1;
	} else if(strcmp(parm,"BOOLCUT")==0 || strcmp(parm,"SECOND_BOOLCUT")==0)
		sscanf(dim,"%f",&var[0].second_BOOLCUT);
	else if(strcmp(parm,"FIRST_BOOLCUT")==0)
		sscanf(dim,"%f",&var[0].first_BOOLCUT);
	else if(strcmp(parm,"SCANSLIDE")==0)
		sscanf(dim,"%d",&var[0].SCANSLIDE);
	else if(strcmp(parm,"SCAN")==0) {
		var[0].SCAN=T_FLAG;
		if(T_FLAG) var[0].PAIRWISE=var[0].TREEWISE=0;
	} else if(strcmp(parm,"SCANMODE")==0) {
	        sscanf(dim,"%d",&var[0].SCANMODE);
		if(var[0].SCANMODE==1) var[0].PAIRALIGN=1; 
	} else if(strcmp(parm,"SCANCUT")==0) 
		sscanf(dim,"%f",&var[0].SCANCUT);
	else if(strcmp(parm,"SECSCREEN")==0)
		var[0].SECSCREEN=T_FLAG;
	else if(strcmp(parm,"CO")==0)
		var[0].CO=T_FLAG;
	else if(strcmp(parm,"SECSCREENMAX")==0)
		sscanf(dim,"%f",&var[0].SECSCREENMAX);
	else if(strcmp(parm,"SCANTRUNC")==0)
		var[0].SCANTRUNC=T_FLAG;
	else if(strcmp(parm,"SCANTRUNCFACTOR")==0)
		sscanf(dim,"%f",&var[0].SCANTRUNCFACTOR);
	else if(strcmp(parm,"DATABASE")==0)
		strcpy(&var[0].database[0],dim);
  	else if(strcmp(parm,"SCANFILE")==0)
		strcpy(&var[0].scanfile[0],dim);
	else if(strcmp(parm,"LOGFILE")==0)
		strcpy(&var[0].logfile[0],dim);
	else if(strcmp(parm,"SECTYPE")==0)
		sscanf(dim,"%d",&var[0].SECTYPE);
	else if(strcmp(parm,"SCANSEC")==0)
		sscanf(dim,"%d",&var[0].SCANSEC);
	else if(strcmp(parm,"SECFILE")==0)
		strcpy(&var[0].secfile[0],dim);
	else if(strcmp(parm,"BOOLEAN")==0)
		var[0].BOOLEAN=T_FLAG;
	else if(strcmp(parm,"BOOLMETHOD")==0)
		sscanf(dim,"%d",&var[0].BOOLMETHOD);
	else if(strcmp(parm,"LISTFILE")==0)
		strcpy(&var[0].listfile[0],dim);
	else if(strcmp(parm,"STAMPDIR")==0)
		strcpy(&var[0].stampdir[0],dim);
	else if(strcmp(parm,"CLUST")==0)
		var[0].CLUST=T_FLAG;
	else if(strcmp(parm,"COLUMNS")==0)
		sscanf(dim,"%d",&var[0].COLUMNS);
	else if(strcmp(parm,"SW")==0)
		sscanf(dim,"%d",&var[0].SW);
	else if(strcmp(parm,"CCFACTOR")==0)
		sscanf(dim,"%f",&var[0].CCFACTOR);
	else if(strcmp(parm,"CCADD")==0)
		var[0].CCADD=T_FLAG;
	else if(strcmp(parm,"MINFIT")==0)
		sscanf(dim,"%d",&var[0].MINFIT);
	else if(strcmp(parm,"ROUGHALIGN")==0)
		strcpy(var[0].roughalign,dim);
	else if(strcmp(parm,"FIRST_THRESH")==0)
		sscanf(dim,"%f",&var[0].first_THRESH);
	else if(strcmp(parm,"MIN_FRAC")==0)
		sscanf(dim,"%f",&var[0].MIN_FRAC);
	else if(strcmp(parm,"SCORERISE")==0)
		var[0].SCORERISE=T_FLAG;
	else if(strcmp(parm,"SKIPAHEAD")==0)
		var[0].SKIPAHEAD=T_FLAG;
	else if(strcmp(parm,"SCANSCORE")==0)
		sscanf(dim,"%d",&var[0].SCANSCORE);
	else if(strcmp(parm,"PAIROUTPUT")==0)
		var[0].PAIROUTPUT=T_FLAG;
	else if(strcmp(parm,"ALLPAIRS")==0)
		var[0].ALLPAIRS=T_FLAG;
	else if(strcmp(parm,"DSSP")==0)
		var[0].DSSP=T_FLAG;
	else if(strcmp(parm,"SLOWSCAN")==0)
		var[0].SLOWSCAN=T_FLAG;
	else if(strcmp(parm,"VERBOSE")==0)
	  	var[0].verbose=T_FLAG;
	else  {
    	    printf("Unrecognised Dimension Command\n");
	    printf("%s %s\n",parm,dim);
	    return -1;
	}
	while((c=getc(fp))!=(char)EOF && c!='\n'); /* read the end of the line, allows for comments */
	if(c==(char)EOF) break;
    }
    if(var[0].SCAN && (var[0].PAIRWISE || var[0].TREEWISE)) {
       fprintf(stderr,"error: cannot specify SCAN and either PAIRWISE or TREEWISE\n");
       return -1;
    }
    
    if((var[0].SCAN) && (var[0].SCANMODE==1)) { 
		var[0].SCANALIGN=1;
		var[0].PAIRALIGN=1;
    }
    if(var[0].SW==1 && var[0].STATS && (var[0].SCAN || var[0].PAIRWISE) ) {
      fprintf(stderr,"error: corner cutting cannot be used in conjunction with STATS ==1\n");
      return -1;
    }
    if(var[0].CLUST!=0 && var[0].CLUST!=1) {
      fprintf(stderr,"error: unrecognized CLUST value, CLUST = %d\n",var[0].CLUST);
      return -1;
    }
    if(var[0].CLUSTMETHOD!=0 && var[0].CLUSTMETHOD!=1) {
      fprintf(stderr,"error: unrecognized CLUSTMETHOD value, CLUSTMETHOD = %d\n",var[0].CLUSTMETHOD); 
      return -1;
    }
    free(parm); free(dim);
    return 0;
}
