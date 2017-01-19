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
#include <math.h>


#define lastmod "25 March 1999"

/* STAMP version 4.2 
 * Lots and lots and lots of changes 
 * RTFM */

void exit_error();
void help_exit_error();

int main(int argc, char *argv[]) {
  
  int i,j/*,k,test*/;
  int ndomain,total,add;
  int gottrans;
  int T_FLAG;
  
/*  char c; */
  char *env;
  char *deffile,*keyword,*value;
  
  FILE *PARMS,*TRANS,*PDB;
  
  struct parameters *parms;
  struct domain_loc *domain;
  
  if(argc<2) exit_error();
  
  /* get environment variable */
  if((env=getenv("STAMPDIR"))==NULL) {
	  fprintf(stderr,"error: environment variable STAMPDIR must be set\n");
	  exit(-1);
  }
  parms=(struct parameters*)malloc(sizeof(struct parameters));
  
  strcpy(parms[0].stampdir,env);
  
  /* read in default parameters from $STAMPDIR/stamp.defaults */
  deffile=(char*)malloc(1000*sizeof(char));
#if defined(_MSC_VER)
  sprintf(deffile,"%s\\stamp.defaults",env);
#else
  sprintf(deffile,"%s/stamp.defaults",env);
#endif
  if((PARMS=fopen(deffile,"r"))==NULL) {
    fprintf(stderr,"error: default parameter file %s does not exist\n",deffile);
    exit(-1);
  }
  if(getpars(PARMS,parms)==-1) exit(-1);
  fclose(PARMS);
  
  /* define DSSP directory file name */
  sprintf(&parms[0].dsspfile[0],"%s/dssp.directories",env);
  
  /* now search the command line for commands */
  keyword=(char*)malloc(1000*sizeof(char));
  value=(char*)malloc(1000*sizeof(char));
  for(i=1; i<argc; ++i) {
    if(argv[i][0]!='-') exit_error();
    strcpy(keyword,&argv[i][1]);
    if(i+1<argc) strcpy(value,argv[i+1]);
    else strcpy(value,"none");
    for(j=0; j<strlen(keyword); ++j) 
      keyword[j]=ltou(keyword[j]); /* change to upper case */
    T_FLAG=(value[0]=='Y' || value[0]=='y' || value[0]=='1' || 
	    value[0]=='T' || value[0]=='t' || value[0]=='o' || 
	    value[0]=='O');
    /* enables one to write '1', 'YES', 'Yes', 'yes', 'T_FLAG', 'True' or 'true' to 
     *  set any boolean parmsiable to one */
    if((strcmp(&argv[i][1],"l")==0) || (strcmp(&argv[i][1],"f")==0) || (strcmp(&argv[i][1],"p")==0)) {
      if(i+1>=argc) exit_error();
      /* listfile name */
      strcpy(parms[0].listfile,argv[i+1]);
      i++;
    } else if(strcmp(&argv[i][1],"P")==0) {
      /* want to read in parameter file */
      if(i+1>=argc) exit_error();
      if((PARMS=fopen(argv[i+1],"r"))==NULL) {
	fprintf(stderr,"error opening file %s\n",argv[i+1]);
	exit(-1);
      }
      if(getpars(PARMS,parms)==-1) exit(-1);
      fclose(PARMS);
      i++;
    } else if(strcmp(&argv[i][1],"o")==0) {
      /* output file */
      if(i+1>=argc) exit_error();
      strcpy(parms[0].logfile,argv[i+1]);
      i++;
    } else if(strcmp(&argv[i][1],"help")==0) {
      help_exit_error();
    } else if((strcmp(&argv[i][1],"V")==0) || (strcmp(&argv[i][1],"v")==0)) {
      parms[0].verbose=1;
      strcpy(parms[0].logfile,"stdout");
    } else if(strcmp(&argv[i][1],"s")==0) {
      parms[0].SCAN=1;
      parms[0].TREEWISE=parms[0].PAIRWISE=0;
    } else if(strcmp(&argv[i][1],"n")==0) {
      if(i+1>=argc) exit_error();
      sscanf(argv[i+1],"%d",&parms[0].NPASS);
      i++;
      if(parms[0].NPASS!=1 && parms[0].NPASS!=2) exit_error();
    } else if(strcmp(keyword,"PAIRPEN") == 0 || strcmp(keyword,"PEN")==0 || strcmp(keyword,"SECOND_PAIRPEN")==0) {
      sscanf(value,"%f",&parms[0].second_PAIRPEN); i++;
    } else if(strcmp(keyword,"FIRST_PAIRPEN")==0) {
      sscanf(value,"%f",&parms[0].first_PAIRPEN); i++;
    } else if(strcmp(keyword,"MAXPITER") == 0 || strcmp(keyword,"MAXSITER") == 0) {
      sscanf(value,"%d",&parms[0].MAXPITER); i++;
    } else if(strcmp(keyword,"MAXTITER") == 0) {
      sscanf(value,"%d",&parms[0].MAXTITER); i++;
    } else if(strcmp(keyword,"TREEPEN") == 0 || strcmp(keyword,"SECOND_TREEPEN")==0) {
      sscanf(value,"%f",&parms[0].second_TREEPEN); i++;
    } else if(strcmp(keyword,"FIRST_TREEPEN")==0) {
      sscanf(value,"%f",&parms[0].first_TREEPEN); i++;
    } else if(strcmp(keyword,"SCORETOL") == 0) {
      sscanf(value,"%f",&parms[0].SCORETOL); i++;
    } else if(strcmp(keyword,"CLUSTMETHOD") == 0) {
      sscanf(value,"%d",&parms[0].CLUSTMETHOD); i++;
    } else if(strcmp(keyword,"E1") == 0 || strcmp(keyword,"SECOND_E1")==0) {
      sscanf(value,"%f",&parms[0].second_E1); i++;
    } else if(strcmp(keyword,"E2") == 0 || strcmp(keyword,"SECOND_E2")==0) {
      sscanf(value,"%f",&parms[0].second_E2); i++;
    } else if(strcmp(keyword,"FIRST_E1")==0) { 
      sscanf(value,"%f",&parms[0].first_E1); i++;
    } else if(strcmp(keyword,"FIRST_E2")==0) {
      sscanf(value,"%f",&parms[0].first_E2); i++;
    } else if(strcmp(keyword,"NPASS")==0) {
      sscanf(value,"%d",&parms[0].NPASS); i++;
      if(parms[0].NPASS!=1 && parms[0].NPASS!=2) {
	fprintf(stderr,"error: NPASS must be either 1 or 2\n");
	return -1;
      }
    } else if(strcmp(keyword,"CUTOFF") == 0 || strcmp(keyword,"SECOND_CUTOFF")==0) {
      sscanf(value,"%f",&parms[0].second_CUTOFF); i++;
    } else if(strcmp(keyword,"FIRST_CUTOFF")==0)  {
      sscanf(value,"%f",&parms[0].first_CUTOFF); i++;
    } else if(strcmp(keyword,"TREEPLOT") == 0) {
      parms[0].TREEPLOT=T_FLAG; i++;
    } else if(strcmp(keyword,"PAIRPLOT") == 0) {
      parms[0].PAIRPLOT=T_FLAG; i++;
    } else if(strcmp(keyword,"NALIGN") == 0) {
      sscanf(value,"%d",&parms[0].NALIGN); i++;
    } else if(strcmp(keyword,"DISPALL") == 0) {
      parms[0].DISPALL=T_FLAG; i++;
    } else if(strcmp(keyword,"HORIZ") ==0) {
      parms[0].HORIZ=T_FLAG; i++;
    } else if(strcmp(keyword,"ADD") ==0) {
      sscanf(value,"%f",&parms[0].ADD); i++;
    } else if(strcmp(keyword,"NMEAN") ==0) {
      sscanf(value,"%f",&parms[0].NMEAN); i++;
    } else if(strcmp(keyword,"NSD") ==0) {
      sscanf(value,"%f",&parms[0].NSD); i++;
    } else if(strcmp(keyword,"STATS") ==0) {
      parms[0].STATS=T_FLAG; i++;
    } else if(strcmp(keyword,"NA") == 0) {
      sscanf(value,"%f",&parms[0].NA); i++;
    } else if(strcmp(keyword,"NB") == 0) {
      sscanf(value,"%f",&parms[0].NB); i++;
    } else if(strcmp(keyword,"NASD") == 0) {
      sscanf(value,"%f",&parms[0].NASD); i++;
    } else if(strcmp(keyword,"NBSD") == 0) {
      sscanf(value,"%f",&parms[0].NBSD); i++;
    } else if(strcmp(keyword,"PAIRWISE") == 0)  {
      parms[0].PAIRWISE=T_FLAG; i++;
    } else if(strcmp(keyword,"TREEWISE") == 0) {
      parms[0].TREEWISE=T_FLAG; i++;
    } else if(strcmp(keyword,"ORDFILE") == 0) {
      strcpy(parms[0].ordfile,value); i++;
    } else if(strcmp(keyword,"TREEFILE") == 0) {
      strcpy(parms[0].treefile,value); i++;
    } else if(strcmp(keyword,"PLOTFILE") == 0) {
      strcpy(parms[0].plotfile,value); i++;
    } else if(strcmp(keyword,"PREFIX") == 0 || strcmp(keyword,"TRANSPREFIX")==0 || strcmp(keyword,"STAMPPREFIX")==0) {
      strcpy(parms[0].transprefix,value); i++;
    } else if(strcmp(keyword,"MATFILE") == 0) {
      strcpy(parms[0].matfile,value); i++;
    } else if(strcmp(keyword,"THRESH") ==0) {
      sscanf(value,"%f",&parms[0].THRESH); i++;
    } else if(strcmp(keyword,"TREEALIGN")==0) {
      parms[0].TREEALIGN=T_FLAG; i++;
    } else if(strcmp(keyword,"TREEALLALIGN")==0) {
      parms[0].TREEALLALIGN=T_FLAG; i++; 
    } else if(strcmp(keyword,"PAIRALIGN")==0 || strcmp(keyword,"SCANALIGN")==0)  {
      parms[0].PAIRALIGN=T_FLAG; i++;
    } else if(strcmp(keyword,"PAIRALLALIGN")==0 || strcmp(keyword,"SCANALLALIGN")==0) {
      parms[0].PAIRALLALIGN=T_FLAG; i++;
    } else if(strcmp(keyword,"PRECISION")==0) {
      sscanf(value,"%d",&parms[0].PRECISION); i++;
    } else if(strcmp(keyword,"MAX_SEQ_LEN")==0) {
      sscanf(value,"%d",&parms[0].MAX_SEQ_LEN); i++;
    } else if(strcmp(keyword,"ROUGHFIT")==0) {
      parms[0].ROUGHFIT=T_FLAG; i++;
    } else if(strcmp(keyword,"ROUGH")==0) {
      parms[0].ROUGHFIT=1;
    } else if(strcmp(keyword,"ROUGHOUT")==0) {
      parms[0].roughout=1;
    } else if(strcmp(keyword,"ROUGHOUTFILE")==0) {
      if(i+1>=argc) exit_error();
      strcpy(&parms[0].roughoutfile[0],argv[i+1]);
      i++;
      parms[0].roughout=1;
    } else if(strcmp(keyword,"BOOLCUT")==0 || strcmp(keyword,"SECOND_BOOLCUT")==0) {
      sscanf(value,"%f",&parms[0].second_BOOLCUT); i++;
    } else if(strcmp(keyword,"FIRST_BOOLCUT")==0) {
      sscanf(value,"%f",&parms[0].first_BOOLCUT); i++;
    } else if(strcmp(keyword,"SCANSLIDE")==0) {
      sscanf(value,"%d",&parms[0].SCANSLIDE); i++;
    } else if(strcmp(keyword,"SCAN")==0) {
      parms[0].SCAN=T_FLAG; i++;
      if(T_FLAG) 
	parms[0].PAIRWISE=parms[0].TREEWISE=0;
    } else if(strcmp(keyword,"SCANMODE")==0) {
      sscanf(value,"%d",&parms[0].SCANMODE); i++;
      if(parms[0].SCANMODE==1) parms[0].PAIRALIGN=1;
    } else if(strcmp(keyword,"SCANCUT")==0)  {
      sscanf(value,"%f",&parms[0].SCANCUT); i++;
    } else if(strcmp(keyword,"SECSCREEN")==0) {
      parms[0].SECSCREEN=T_FLAG; i++;
    } else if(strcmp(keyword,"SECSCREENMAX")==0) {
      sscanf(value,"%f",&parms[0].SECSCREENMAX); i++;
    } else if(strcmp(keyword,"SCANTRUNC")==0) {
      parms[0].SCANTRUNC=T_FLAG; i++;
    } else if(strcmp(keyword,"SCANTRUNCFACTOR")==0) {
      sscanf(value,"%f",&parms[0].SCANTRUNCFACTOR); i++;
    } else if(strcmp(keyword,"DATABASE")==0) {
      strcpy(&parms[0].database[0],value); i++;
    } else if(strcmp(keyword,"SCANFILE")==0) {
      strcpy(&parms[0].scanfile[0],value); i++;
    } else if(strcmp(keyword,"LOGFILE")==0) {
      strcpy(&parms[0].logfile[0],value); i++;
    } else if(strcmp(keyword,"SECTYPE")==0) {
      sscanf(value,"%d",&parms[0].SECTYPE); i++;
    } else if(strcmp(keyword,"SCANSEC")==0) {
      sscanf(value,"%d",&parms[0].SCANSEC); i++;
    } else if(strcmp(keyword,"SECFILE")==0) {
      strcpy(&parms[0].secfile[0],value); i++;
      parms[0].SECTYPE=2;
    } else if(strcmp(keyword,"BOOLEAN")==0) {
      parms[0].BOOLEAN=T_FLAG; i++;
    } else if(strcmp(keyword,"BOOLMETHOD")==0) {
      sscanf(value,"%d",&parms[0].BOOLMETHOD); i++;
    } else if(strcmp(keyword,"LISTFILE")==0) {
      strcpy(&parms[0].listfile[0],value); i++;
    } else if(strcmp(keyword,"STAMPDIR")==0) {
      strcpy(&parms[0].stampdir[0],value); i++;
    } else if(strcmp(keyword,"CLUST")==0) {
      parms[0].CLUST=T_FLAG; i++;
    } else if(strcmp(keyword,"COLUMNS")==0) {
      sscanf(value,"%d",&parms[0].COLUMNS); i++;
    } else if(strcmp(keyword,"SW")==0) {
      sscanf(value,"%d",&parms[0].SW); i++;
    } else if(strcmp(keyword,"CCFACTOR")==0) {
      sscanf(value,"%f",&parms[0].CCFACTOR); i++;
    } else if(strcmp(keyword,"CCADD")==0) {
      parms[0].CCADD=T_FLAG; i++;
    } else if(strcmp(keyword,"MINFIT")==0) {
      sscanf(value,"%d",&parms[0].MINFIT); i++;
    } else if(strcmp(keyword,"ROUGHALIGN")==0) {
      strcpy(parms[0].roughalign,value); i++;
    } else if(strcmp(keyword,"FIRST_THRESH")==0) {
      sscanf(value,"%f",&parms[0].first_THRESH); i++;
    } else if(strcmp(keyword,"MIN_FRAC")==0) {
      sscanf(value,"%f",&parms[0].MIN_FRAC); i++;
    } else if(strcmp(keyword,"SCORERISE")==0) {
      parms[0].SCORERISE=T_FLAG; i++;
    } else if(strcmp(keyword,"SKIPAHEAD")==0) {
      parms[0].SKIPAHEAD=T_FLAG; i++;
    } else if(strcmp(keyword,"SCANSCORE")==0) {
      sscanf(value,"%d",&parms[0].SCANSCORE); i++;
    } else if(strcmp(keyword,"PAIROUTPUT")==0) {
      parms[0].PAIROUTPUT=T_FLAG; i++;
    } else if(strcmp(keyword,"ALLPAIRS")==0) {
      parms[0].ALLPAIRS=T_FLAG; i++;
    } else if (strcmp(keyword,"ATOMTYPE")==0) {
      parms[0].ATOMTYPE=T_FLAG; i++;
    } else if(strcmp(keyword,"DSSP")==0) {
      parms[0].DSSP=T_FLAG; i++;
    } else if(strcmp(keyword,"SLOWSCAN")==0) {
      parms[0].SLOWSCAN=T_FLAG; i++;
    } else if(strcmp(keyword,"SLOW")==0) {
      parms[0].SLOWSCAN=1;
    } else if(strcmp(keyword,"CUT")==0) {
      parms[0].CO=1;
    } else if(strcmp(&argv[i][1],"slide")==0) {
      if(i+1>=argc) exit_error();
      sscanf(argv[i+1],"%d",&parms[0].SCANSLIDE); 
      i++;
    } else if(strcmp(&argv[i][1],"d")==0) {
      /* database file */
      if(i+1>=argc) exit_error();
      strcpy(&parms[0].database[0],argv[i+1]);
      i++;
    } else if(strcmp(&argv[i][1],"pen1")==0) {
      if(i+1>=argc) exit_error();
      sscanf(argv[i+1],"%f",&parms[0].first_PAIRPEN);
      i++;
    } else if(strcmp(&argv[i][1],"pen2")==0) {
      if(i+1>=argc) exit_error();
      sscanf(argv[i+1],"%f",&parms[0].second_PAIRPEN);
      i++;
    } else if(strcmp(&argv[i][1],"prefix")==0) {
      if(i+1>=argc) exit_error();
      strcpy(&parms[0].transprefix[0],argv[i+1]);
      i++;
    } else if(strcmp(&argv[i][1],"scancut")==0) {
      if(i+1>=argc) exit_error();
      sscanf(argv[i+1],"%f",&parms[0].SCANCUT);
      i++;
    } else if(strcmp(&argv[i][1],"opd")==0) {
      parms[0].opd=1;
    } else  {
      exit_error();
    }
  }
  free(keyword); 
  free(value);
  
  
  /* make the names of all the output files using the prefix */
  sprintf(&parms[0].ordfile[0],"%s.ord",parms[0].transprefix);
  sprintf(&parms[0].treefile[0],"%s.tree",parms[0].transprefix);
  sprintf(&parms[0].plotfile[0],"%s.plot",parms[0].transprefix);
  sprintf(&parms[0].matfile[0],"%s.mat",parms[0].transprefix);
  sprintf(&parms[0].roughalign[0],"%s_align.rough",parms[0].transprefix);
  sprintf(&parms[0].scanfile[0],"%s.scan",parms[0].transprefix);
  
  if(strcmp(parms[0].logfile,"stdout")==0 || 
     strcmp(parms[0].logfile,"STDOUT")==0) {
    parms[0].LOG=stdout;
  } else if(strcmp(parms[0].logfile,"silent")==0 ||
	    strcmp(parms[0].logfile,"SILENT")==0) {
#if defined(_MSC_VER)
    parms[0].LOG=stdout;
#else
    parms[0].LOG=fopen("/dev/null","w");
#endif
  } else {
    if((parms[0].LOG=fopen(parms[0].logfile,"w"))==NULL) {
      fprintf(stderr,"error opening file %s\n",parms[0].logfile);
      exit(-1);
    } 
  }
  
  if(strcmp(parms[0].logfile,"silent")==0) {
    printf("\nSTAMP Structural Alignment of Multiple Proteins\n");
    printf(" by Robert B. Russell & Geoffrey J. Barton \n");
    printf(" Please cite PROTEINS, v14, 309-323, 1992\n\n");
  }
  fprintf(parms[0].LOG,"-------------------------------------------------------------------------------\n");
  fprintf(parms[0].LOG,"                                   S t A M P\n");
  fprintf(parms[0].LOG,"                             Structural Alignment of\n");
  fprintf(parms[0].LOG,"                               Multiple Proteins\n");
  fprintf(parms[0].LOG,"                     By Robert B. Russell & Geoffrey J. Barton \n");
  fprintf(parms[0].LOG,"                       Last Modified: %s\n",lastmod);
  fprintf(parms[0].LOG,"         Please cite Ref: Russell and GJ Barton, PROTEINS, v14, 309-323, 1992\n");
  fprintf(parms[0].LOG,"-------------------------------------------------------------------------------\n\n");
  
  
  fprintf(parms[0].LOG,"STAMPDIR has been set to %s\n\n\n",parms[0].stampdir);
  /* read in coordinate locations and initial transformations */
  if((TRANS = fopen(parms[0].listfile,"r")) == NULL) {
    fprintf(stderr,"error: file %s does not exist\n",parms[0].listfile);
    exit(-1);
  }
  /* determine the number of domains specified */
  ndomain=count_domain(TRANS);
  domain=(struct domain_loc*)malloc(ndomain*sizeof(struct domain_loc));
  rewind(TRANS);
  if(getdomain(TRANS,domain,&ndomain,ndomain,&gottrans,parms[0].stampdir,parms[0].DSSP,parms[0].LOG)==-1) exit(-1);
  fclose(TRANS);
  
  fprintf(parms[0].LOG,"Details of this run:\n");
  if(parms[0].PAIRWISE) fprintf(parms[0].LOG,"PAIRWISE mode specified\n");
  if(parms[0].TREEWISE) fprintf(parms[0].LOG,"TREEWISE mode specified\n");
  if(parms[0].SCAN) fprintf(parms[0].LOG,"SCAN mode specified\n");
  
  if(!parms[0].SCAN) {
    /* if no MINFIT has been given, then take the smallest length and divide it by two */
    if(parms[0].MINFIT==-1) {
      parms[0].MINFIT=parms[0].MAXLEN;
      for(i=0; i<ndomain; ++i) if(domain[i].ncoords<parms[0].MINFIT) parms[0].MINFIT=domain[i].ncoords;
      parms[0].MINFIT/=2;
    }
    fprintf(parms[0].LOG,"  pairwise score file: %s\n",parms[0].matfile);
    if(parms[0].TREEWISE) {
      fprintf(parms[0].LOG,"  tree order file: %s\n",parms[0].ordfile);
      fprintf(parms[0].LOG,"  tree file: %s\n",parms[0].treefile); 
      fprintf(parms[0].LOG,"  tree plot file: %s\n",parms[0].plotfile);
    }
  } else {
    fprintf(parms[0].LOG,"   SCANMODE set to %d\n",parms[0].SCANMODE);
    fprintf(parms[0].LOG,"   SCANSCORE set to %d\n",parms[0].SCANSCORE);
    fprintf(parms[0].LOG,"    (see documentation for an explanation)\n");
    if(parms[0].opd==1) fprintf(parms[0].LOG,"   Domains will be skipped after the first match is found\n");
    if(parms[0].SCANMODE==1) {
      fprintf(parms[0].LOG,"     Transformations for Sc values greater than %f are to be output\n",parms[0].SCANCUT);
      fprintf(parms[0].LOG,"     to the file %s\n",parms[0].transprefix);
    } else {
      fprintf(parms[0].LOG,"     Only the scores are to be output to the file %s\n",parms[0].scanfile);
    }
    fprintf(parms[0].LOG,"  secondary structures are ");
    switch(parms[0].SCANSEC) {
    case 0: fprintf(parms[0].LOG," not to be considered\n"); break;
    case 1: fprintf(parms[0].LOG," to be from DSSP\n"); break;
    case 2: fprintf(parms[0].LOG," to be read in from %s\n",parms[0].secfile); break;
    default: fprintf(parms[0].LOG," not to be considered\n"); 
    }
    if(parms[0].SECSCREEN) {
      fprintf(parms[0].LOG,"   An initial screen on secondary structure content is to performed when possible\n");
      fprintf(parms[0].LOG,"   Secondary structure summaries farther than %6.2f %% apart result in\n",parms[0].SECSCREENMAX);
      fprintf(parms[0].LOG,"     a comparison being ignored\n");
    }
    fprintf(parms[0].LOG,"   Initial fits are to be performed by aligning the N-terminus of the query\n    with every %d residue of the database sequence\n",parms[0].SCANSLIDE);
    fprintf(parms[0].LOG,"    of the query along the database structure.\n");
    if(parms[0].SCANTRUNC) {
      fprintf(parms[0].LOG,"   If sequences in the database are > %5.3f x the query sequence length\n",parms[0].SCANTRUNCFACTOR);
      fprintf(parms[0].LOG,"    then a fraction of the the database structure, corresponding to this\n");
      fprintf(parms[0].LOG,"    of length %5.3f x the query, will be considered\n",parms[0].SCANTRUNCFACTOR);
      fprintf(parms[0].LOG,"   comparisons are to be ignored if the database structure is less than\n    %6.4f x the length of the query structure\n",parms[0].MIN_FRAC);
    }
    fprintf(parms[0].LOG,"   Domain database file to be scanned %s\n",parms[0].database);
  }
  
  if(parms[0].TREEWISE) 
    fprintf(parms[0].LOG,"  output files prefix: %s\n",parms[0].transprefix);
  
  fprintf(parms[0].LOG,"\n\nParameters:\n");
  fprintf(parms[0].LOG,"Rossmann and Argos parameters:\n");
  if(parms[0].NPASS==2) {
    fprintf(parms[0].LOG,"  Two fits are to be performed, the first fit with:\n");
    fprintf(parms[0].LOG,"   E1=%7.3f,",parms[0].first_E1);
    fprintf(parms[0].LOG," E2=%7.3f,",parms[0].first_E2);
    fprintf(parms[0].LOG," CUT=%7.3f,",parms[0].first_CUTOFF);
    fprintf(parms[0].LOG," PAIRPEN=%7.3f,",parms[0].first_PAIRPEN);
    fprintf(parms[0].LOG," TREEPEN=%7.3f\n",parms[0].first_TREEPEN);
    /*	   fprintf(parms[0].LOG,"   E1=%7.3f, E2=%7.3f, CUT=%7.3f, PAIRPEN=%7.3f, TREEPEN=%7.3f\n",
	   parms[0].first_E1,parms[0].first_E2,parms[0].first_CUTOFF,parms[0].first_PAIRPEN,parms[0].first_TREEPEN); */
    fprintf(parms[0].LOG,"  The second fit with:\n");
  } else 
    fprintf(parms[0].LOG,"  One fit is to performed with:\n");
  fprintf(parms[0].LOG,"   E1=%7.3f, E2=%7.3f, CUT=%7.3f, PAIRPEN=%7.3f, TREEPEN=%7.3f\n",
	  parms[0].second_E1,parms[0].second_E2,parms[0].second_CUTOFF,parms[0].second_PAIRPEN,parms[0].second_TREEPEN);
  if(parms[0].BOOLEAN) {
    fprintf(parms[0].LOG,"  BOOLEAN mode specified\n");
    fprintf(parms[0].LOG,"  A boolean matrix will be calculated corresponding to whether\n");
    fprintf(parms[0].LOG,"   positions have Pij values greater than:\n");
    if(parms[0].NPASS==2)
      fprintf(parms[0].LOG,"    %7.3f, for the first fit and\n",parms[0].first_BOOLCUT);
    fprintf(parms[0].LOG,"    %7.3f",parms[0].second_BOOLCUT);
    if(parms[0].NPASS==2) 
      fprintf(parms[0].LOG," for the second fit.\n");
    else
      fprintf(parms[0].LOG,".\n");
    fprintf(parms[0].LOG,"  In the multiple case, this criteria must be satisfied for *all*\n");
    fprintf(parms[0].LOG,"   possible pairwise comparisons\n");
  }
  if(parms[0].SW==1) {
    fprintf(parms[0].LOG,"  Corner cutting is to be performed\n");
    fprintf(parms[0].LOG,"    Corner cutting length: %6.2f\n",parms[0].CCFACTOR);
    if(parms[0].CCADD) 
      fprintf(parms[0].LOG,"    The length difference is to be added to this value\n");
  } else {
    fprintf(parms[0].LOG,"  The entire SW matrix is to be calculated and used\n");
  }
  fprintf(parms[0].LOG,"  The minimum length of alignment to be evaluated further is %3d residues\n",parms[0].MINFIT);
  fprintf(parms[0].LOG,"\n");
  fprintf(parms[0].LOG,"  Convergence tolerance SCORETOL= %f %%\n", parms[0].SCORETOL);
  fprintf(parms[0].LOG,"  Other parameters:\n");
  fprintf(parms[0].LOG,"    MAX_SEQ_LEN=%d, MAXPITER=%d, MAXTITER=%d\n", 
	  parms[0].MAX_SEQ_LEN,parms[0].MAXPITER,parms[0].MAXTITER);
  fprintf(parms[0].LOG,"    PAIRPLOT (SCANPLOT) = %d, TREEPLOT = %d, PAIRALIGN (SCANALIGN) = %d, TREEALIGN = %d\n",
	  parms[0].PAIRPLOT,parms[0].TREEPLOT,parms[0].PAIRALIGN,parms[0].TREEALIGN);
  fprintf(parms[0].LOG,"    PAIRALLALIGN (SCANALLALIGN) = %d, TREEALLALIGN = %d\n",parms[0].PAIRALLALIGN,parms[0].TREEALLALIGN);
  
  if(!parms[0].BOOLEAN) {
    fprintf(parms[0].LOG,"\n\nDetails of Confidence value calculations:\n");
    if(parms[0].STATS) fprintf(parms[0].LOG,"  actual mean and standard deviations are to be\n   used for determination of Pij' values.\n");
    else {
      fprintf(parms[0].LOG,"  pre-set mean and standard deviations are to be used\n   and multiple comparisons are to be corrected.\n");
      fprintf(parms[0].LOG,"  mean Xt = %f, standard deviation SDt = %f\n", parms[0].NMEAN,parms[0].NSD);
      fprintf(parms[0].LOG,"  for the multiple case:\n");
      fprintf(parms[0].LOG,"    pairwise means are to be calculated from:\n      Xp = exp(%6.4f * log(length) + %6.4f)\n",parms[0].NA,parms[0].NB);
      fprintf(parms[0].LOG,"     and pairwise standard deviations from:\n     SDp = exp(%6.4f * log(length) + %6.4f)\n",parms[0].NASD,parms[0].NBSD);
      fprintf(parms[0].LOG,"    the mean to be used is calculated from:  \n      Xc =  (Xm/Xp) * Xt).\n");
      fprintf(parms[0].LOG,"     and the standard deviation from: \n     SDc = (SDm/SDp)*SDt).\n");
    } /* End of if(parms[0].STATS) */
  } else {
    fprintf(parms[0].LOG,"  Positional values will consist of one's or zeros depending on whether\n");
    fprintf(parms[0].LOG,"   a position satisfies the BOOLEAN criterion above\n");
    fprintf(parms[0].LOG,"  The score (Sp) for each alignment will be a sum of these positions.\n");
  } /* end of if(parms[0].BOOLEAN */
  
  if(!parms[0].SCAN && parms[0].TREEWISE) {
    fprintf(parms[0].LOG,"\n\nTree is to be generated by ");
    if(parms[0].CLUSTMETHOD==0) fprintf(parms[0].LOG,"1/rms values.\n");
    if(parms[0].CLUSTMETHOD==1) { 
      fprintf(parms[0].LOG,"scores from path tracings modified as follows:\n"); 
      fprintf(parms[0].LOG,"    Sc = (Sp/Lp) * ((Lp-ia)/La) * ((Lp-ib)/Lb),\n");
      fprintf(parms[0].LOG,"    where Sp is the actual score, Lp is the path length.\n");
      fprintf(parms[0].LOG,"    and La & Lb are the lengths of the structures considered.\n");
    } /* End of if(parms[0].METHOD==2) */
  }
  fprintf(parms[0].LOG,"\n\n");

  fprintf(parms[0].LOG,"Reading coordinates...\n");
  for(i=0; i<ndomain; ++i) {
    fprintf(parms[0].LOG,"Domain %3d %s %s\n   ",i+1,domain[i].filename,domain[i].id);
    if((PDB=openfile(domain[i].filename,"r"))==NULL) {
      fprintf(stderr,"error opening file %s\n",domain[i].filename);
      exit(-1);
    }
    domain[i].ncoords=0;
    domain[i].coords=(int**)malloc(parms[0].MAX_SEQ_LEN*sizeof(int*));
    domain[i].aa=(char*)malloc((parms[0].MAX_SEQ_LEN+1)*sizeof(char)); 
    domain[i].numb=(struct brookn*)malloc((parms[0].MAX_SEQ_LEN)*sizeof(struct brookn));
    total=0;
    fprintf(parms[0].LOG,"    ");
    for(j=0; j<domain[i].nobj; ++j) {
      if(!parms[0].DSSP) {
	if(igetca(PDB,&domain[i].coords[total],&domain[i].aa[total],&domain[i].numb[total],
		  &add,domain[i].start[j],domain[i].end[j],domain[i].type[j],(parms[0].MAX_SEQ_LEN-total),
		  domain[i].reverse[j],parms[0].PRECISION,parms[0].ATOMTYPE,parms[0].LOG)==-1) {
	  fprintf(stderr,"Error in domain %s object %d \n",domain[i].id,j+1);
	  exit(-1);
	}
      } else {
	if(igetcadssp(PDB,&domain[i].coords[total],&domain[i].aa[total],&domain[i].numb[total],
		      &add,domain[i].start[j],domain[i].end[j],domain[i].type[j],(parms[0].MAX_SEQ_LEN-total),
		      domain[i].reverse[j],parms[0].PRECISION,parms[0].LOG)==-1) exit(-1);
      }
	  switch(domain[i].type[j]) {
	  case 1: fprintf(parms[0].LOG," all residues"); break;
	  case 2: fprintf(parms[0].LOG," chain %c",domain[i].start[j].cid); break;
	  case 3: fprintf(parms[0].LOG," from %c %4d %c to %c %4d %c",
			  domain[i].start[j].cid,domain[i].start[j].n,domain[i].start[j].in,
			  domain[i].end[j].cid,domain[i].end[j].n,domain[i].end[j].in); break;
	  }
	  fprintf(parms[0].LOG,"%4d CAs ",add);
	  total+=add;
	  closefile(PDB,domain[i].filename); PDB=openfile(domain[i].filename,"r");
    }
    domain[i].ncoords=total;
    fprintf(parms[0].LOG,"=> %4d CAs in total\n",domain[i].ncoords);
    fprintf(parms[0].LOG,"Applying the transformation... \n");
    printmat(domain[i].R,domain[i].V,3,parms[0].LOG);
    fprintf(parms[0].LOG,"      ...to these coordinates.\n");
    matmult(domain[i].R,domain[i].V,domain[i].coords,domain[i].ncoords,parms[0].PRECISION);
    closefile(PDB,domain[i].filename);
  }
  fprintf(parms[0].LOG,"\n\n");
  fprintf(parms[0].LOG,"Secondary structure...\n");
  for(i=0; i<ndomain; ++i) 
    domain[i].sec=(char*)malloc(parms[0].MAX_SEQ_LEN*sizeof(char));
  
  switch(parms[0].SECTYPE) {
  case 0: {
    fprintf(parms[0].LOG,"No secondary structure assignment will be considered\n");
    for(i=0; i<ndomain; ++i) {
      for(j=0; j<domain[i].ncoords; ++j) domain[i].sec[j]='?';
      domain[i].sec[j]='\0';
    }
    parms[0].SECSCREEN=0;
  } break;
  case 1: {
    fprintf(parms[0].LOG,"Will try to find Kabsch and Sander DSSP assignments\n"); 
    
    if(getks(domain,ndomain,parms)!=0) parms[0].SECSCREEN=0;
  } break;
  case 2: {
    fprintf(parms[0].LOG,"Reading in secondary structure assignments from file: %s\n",parms[0].secfile);
    if(getsec(domain,ndomain,parms)!=0) parms[0].SECSCREEN=0;
  } break;
  default: {
    fprintf(stderr,"error: unrecognised secondary structure assignment option\n");
    exit(-1);
  }
  }
  
  fprintf(parms[0].LOG,"\n\n");
  if(parms[0].SCAN) {
    i=0;
    fprintf(parms[0].LOG,"Scanning with domain %s\n",&(domain[i].id[0]));
    if(strcmp(parms[0].logfile,"silent")==0) {
      
      printf("Results of scan will be written to file %s\n",parms[0].scanfile);
      printf("Fits  = no. of fits performed, Sc = STAMP score, RMS = RMS deviation\n");
      printf("Align = alignment length, Nfit = residues fitted, Eq. = equivalent residues\n");
      printf("Secs  = no. equiv. secondary structures, %%I = seq. identity, %%S = sec. str. identity\n");
      printf("P(m)  = P value (p=1/10) calculated after Murzin (1993), JMB, 230, 689-694\n");
      printf("\n");
      printf("     Domain1         Domain2          Fits  Sc      RMS   Len1 Len2 Align Fit   Eq. Secs    %%I    %%S     P(m)\n");
    }
    
    if(parms[0].SLOWSCAN==1) {
      if(slow_scan(domain[i],parms)==-1) exit(-1); 
    } else {
      if(scan(domain[i],parms)==-1) exit(-1);
    }
    if(strcmp(parms[0].logfile,"silent")==0) 
      printf("See the file %s.scan\n",parms[0].transprefix);
    fprintf(parms[0].LOG,"\n");
  } else {
    if(parms[0].ROUGHFIT) if(roughfit(domain,ndomain,parms)==-1) exit(-1);
    if(parms[0].PAIRWISE) if(pairwise(domain,ndomain,parms)==-1) exit(-1);
    if(parms[0].TREEWISE) if(treewise(domain,ndomain,parms)==-1) exit(-1);
  } /* end of if(parms[0].SCAN... */
  
  /* freeing memory to keep purify happy */
  /*
    for(i=0; i<ndomain; ++i) {
    free(domain[i].aa);
    free(domain[i].sec);
    free(domain[i].v); free(domain[i].V);
    for(j=0; j<3; ++j) {
    free(domain[i].R[j]);
    free(domain[i].r[j]);
    }
    free(domain[i].R); 
    free(domain[i].r);
    for(j=0; j<domain[i].ncoords; ++j) 
    free(domain[i].coords[j]);
    free(domain[i].coords);
    free(domain[i].type);
    free(domain[i].start);
    free(domain[i].end);
    free(domain[i].reverse);
    free(domain[i].numb);
    }
  */
  free(domain);
  
  exit(0);
}


void help_exit_error() {
  fprintf(stderr,"format: stamp -f <starting domain file> [options] \n");
  fprintf(stderr,"\n");
  fprintf(stderr,"GENERAL OPTIONS:\n");
  fprintf(stderr,"     -o <output log file>  DEF is stdout\n");
  fprintf(stderr,"     -P <parameter file>   DEF is $STAMPDIR/stamp.defaults\n");
  fprintf(stderr,"     -n <1 or 2 fits>      DEF is 1\n");
  fprintf(stderr,"     -prefix <string>      DEF is stamp_trans\n");
  fprintf(stderr,"     -pen1 <gap penatly 1> DEF is 0\n");
  fprintf(stderr,"     -pen2 <gap penalty 2> DEF is 0\n");
  fprintf(stderr,"     -<parameter> <value> (any parameter defined in the manual)\n");
  fprintf(stderr,"MULTIPLE ALIGNMENT:\n");
  fprintf(stderr,"     -rough => do ROUGHFIT initial superimposition\n");
  fprintf(stderr,"SCANNING:\n");
  fprintf(stderr,"     -s     => invoke SCAN mode\n");
  fprintf(stderr,"     -cut   => truncate domains\n");
  fprintf(stderr,"     -d <database file>              DEF is dbase.dom\n");
  fprintf(stderr,"     -slide <slide parameter>        DEF is 10\n");
  fprintf(stderr,"     -scancut <scan Sc cutoff value> DEF is 2.0 \n");
  exit(-1);
}


void exit_error() {
  fprintf(stderr,"format: stamp -f <starting domain file> [options] \n");
  fprintf(stderr,"        stamp -help  will give a list of options\n");
  exit(-1);
}
