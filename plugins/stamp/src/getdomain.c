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
#include <string.h>
#include <stamp.h>

/* Given a file containing a list of protein descriptors, returns
 *  a list of brookhaven starts and ends, or appropriate wild cards
 *  for subsequent use 
 *
 * New version to remove all the stupid wonk bugs that this used to contain 
 * it now seems very resilient to the placement of newlines, junk, etc */


int getdomain(FILE *IN, struct domain_loc *domains, int *ndomain, int maxdomain, 
	int *gottrans, char *env, int DSSP, FILE *OUTPUT) {

	int i,j;
/*	int comment; */
	int count,end/*,nobjects*/;

/*	char c; */
/*	char *buff; */



	count=0;
	end=0;
	(*gottrans)=0;
	while(!end) {
	  end=domdefine(&domains[count],&i,env,DSSP,IN,OUTPUT);
/*	  printf("Domain %s\n",domains[count].id);
          printdomain(stdout,domains[count],1); */

	  if(i==1) (*gottrans)=1;
	  if(end==-1) {
	     fprintf(stderr,"error in domain specification file\n");
	     return -1;
	  }
	  count+=(!end);
	  if(count>maxdomain && end!=1) {
	    fprintf(stderr,"error have exceeded maximum domain limit\n");
	    return -1;
	  }
	  if(count==maxdomain) end=1;
	} /* end while(!end)... */
	(*ndomain)=count;

	/* check for duplication */
	for(i=0; i<(*ndomain); ++i) 
	   for(j=i+1; j<(*ndomain); ++j)  
	      if(strcmp(domains[i].id,domains[j].id)==0) {
		 fprintf(stderr,"error: domain identifiers must not be the same\n");
		 fprintf(stderr,"       found two copies of %s, domains %d & %d\n",
			domains[i].id,i+1,j+1);
		 return -1; 
	      }

	return 0;
}

int domdefine(struct domain_loc *domain, int *gottrans, char *env, int DSSP, FILE *INPUT, FILE *OUTPUT)

/* reads in the next domain descriptor from a supplied input file 
 * returns 0 if all is well, -1 if an error occurs, 1 if EOF occurs */
{

	int i,j/*,k*/;
	int nobjects;
/*	int comment; */
	int pt;
	int indom;

/* 	char c; */
	char *descriptor;
	char *buff,*guff;
	char *temp;
	char *dirfile;
	
	FILE *TEST;

	buff=(char*)malloc(2000*sizeof(char));
	guff=(char*)malloc(2000*sizeof(char));
	dirfile=(char*)malloc(500*sizeof(char));
	descriptor=(char*)malloc(2000*sizeof(char));
	(*gottrans)=0;
	
	if(DSSP) {
	   sprintf(dirfile,"%s/dssp.directories",env);
	} else {
	   sprintf(dirfile,"%s/pdb.directories",env);
	}

	/* Now then, the best way to do this is to skip the comments, and
	 *  then just read in a buff starting after the last newline, and
	 *  ending at the end brace */

	indom = 0;
	buff[0]='\0';
	while(fgets(guff,1999,INPUT)!=NULL) {
	     if((guff[0]!='%') && (guff[0]!='#')) {
		if(strstr(guff,"{")!=NULL) { indom=1; }
		if(indom) {
			sprintf(&buff[strlen(buff)],"%s",guff);
		}
		if(strstr(guff,"}")!=NULL) { indom=0; break; }
	     }
	}
		 
/*	printf("Domain is %s\n",buff);      */

	/* First read the file name */

	sscanf(buff,"%s",&domain[0].filename[0]); /* read the filename */
	pt=0;
	if((pt=skiptononspace(buff,pt))==-1) getdomain_error(buff);
	sscanf(&buff[pt],"%s",&domain[0].id[0]);	/* read the identifier */
/*	printf("Read in file %s\n",domain[0].filename); */

	/* check to see whether the file exists, otherwise, look for a file
	 *  with a similar ID */
	if((TEST=fopen(domain[0].filename,"r"))==NULL) {
	  /* look for the file */
	  temp=getfile(domain[0].id,dirfile,4,OUTPUT); /* assume the first four characters are the id */
	  if(temp[0]=='\0') {
	    fprintf(OUTPUT,"file for %s not found, nor was any corresponding file\n",domain[0].id);
	    fprintf(OUTPUT,"   found in %s\n",dirfile);
	    free(buff); free(dirfile); free(descriptor);
	    free(guff);
	    return -1; 
	  } else {
	    strcpy(&domain[0].filename[0],temp);
	  }
	  free(temp);
	} else {
	  fclose(TEST);
	}
	   

/*	printf("Updated file and id %s and %s\n",domain[0].filename,domain[0].id); 
	printf("Buff is %s\n",buff); 
	printf("Length is %d\n",strlen(buff)); */

	if((pt=skiptononspace(buff,pt))==-1) getdomain_error(buff);
	/* copy the bit between the braces into the string called descriptor */
	i=0; 
	while(buff[pt]!='{' && buff[pt]!='\n' && buff[pt]!='\0') pt++;
	if(buff[pt]=='\n' || buff[pt]=='\0') getdomain_error(buff);
	pt++;
	if(buff[pt]=='\n' || buff[pt]=='\0') getdomain_error(buff);
	j=0; while(buff[pt]!='}' && buff[pt]!='\0') {
	   if(buff[pt]=='\0') getdomain_error(buff);
	   descriptor[j]=buff[pt];
	   pt++; 
	   j++;
	}
	descriptor[j]='\0';
/*	printf("descriptor= '%s'\n",descriptor);     */
	/* allocation of memory, initially */
	domain[0].reverse=(int*)malloc(sizeof(int));
	domain[0].type=(int*)malloc(sizeof(int));
	domain[0].start=(struct brookn*)malloc(sizeof(struct brookn));
	domain[0].end=(struct brookn*)malloc(sizeof(struct brookn)); 
	domain[0].V=(float*)malloc(3*sizeof(float));
	domain[0].v=(float*)malloc(3*sizeof(float));
	domain[0].R=(float**)malloc(3*sizeof(float*));
	domain[0].r=(float**)malloc(3*sizeof(float*));
	for(i=0; i<3; ++i) {
	   domain[0].R[i]=(float*)malloc(3*sizeof(float));
	   domain[0].r[i]=(float*)malloc(3*sizeof(float));
	   for(j=0; j<3; ++j) 
	     if(i==j) domain[0].R[i][j]=domain[0].r[i][j]=1.0;
	     else domain[0].R[i][j]=domain[0].r[i][j]=0.0;
	     domain[0].V[i]=domain[0].v[i]=0.0;
	}

	nobjects=0;
	for(i=0; i<strlen(descriptor); ++i) descriptor[i]=ltou(descriptor[i]);
	pt=0;
	if(strlen(descriptor)==0) getdomain_error(buff);
	while(descriptor[pt]==' ' && descriptor[pt]!='\0') pt++;
	if(descriptor[pt]=='\0' || descriptor[pt]=='}') getdomain_error(buff);
	while(pt!=-1 && descriptor[pt]!='\0' && descriptor[pt]!='\n') { /* read until end of string */
	   if(strncmp(&descriptor[pt],"REVERSE",7)==0) { /* coordinates are to be reversed */
		 domain[0].reverse[nobjects]=1;
		 pt=skiptononspace(descriptor,pt);
	    } else {
		 domain[0].reverse[nobjects]=0;
		 /* don't skip over the text if the word "REVERSE" isn't there */
	   }
	   if(strncmp(&descriptor[pt],"ALL",3)==0) {  /* want all the coordinates in the file */
		 domain[0].type[nobjects]=1;
		 domain[0].start[nobjects].cid=domain[0].start[nobjects].in=
		     domain[0].end[nobjects].cid=domain[0].end[nobjects].in='?';
	 	 domain[0].start[nobjects].n=domain[0].end[nobjects].n=0;
		 pt=skiptononspace(descriptor,pt);
		 nobjects++;
	   } else if(strncmp(&descriptor[pt],"CHAIN",5)==0) { /* want specific chain only */
		 domain[0].type[nobjects]=2;
		 if((pt=skiptononspace(descriptor,pt))==-1) getdomain_error(buff); /* no chain given */
		 domain[0].start[nobjects].cid=domain[0].end[nobjects].cid=descriptor[pt];
		 domain[0].start[nobjects].in=domain[0].end[nobjects].in='?';
		 domain[0].start[nobjects].n=domain[0].end[nobjects].n=0;
		 pt=skiptononspace(descriptor,pt);
		 nobjects++;
	   } else { /* assume that otherwise a specific start and end will be provided */
		 domain[0].type[nobjects]=3;
		 /* cid 1 */
	 	 if(descriptor[pt]=='_') domain[0].start[nobjects].cid=' ';
		 else domain[0].start[nobjects].cid=descriptor[pt];
		 if((pt=skiptononspace(descriptor,pt))==-1) getdomain_error(buff); 
		 /* n 1 */
		 sscanf(&descriptor[pt],"%d",&domain[0].start[nobjects].n);
		 if((pt=skiptononspace(descriptor,pt))==-1) getdomain_error(buff); 
		 /* ins 1 */
		 if(descriptor[pt]=='_') domain[0].start[nobjects].in=' ';
		 else domain[0].start[nobjects].in=descriptor[pt];
		 if((pt=skiptononspace(descriptor,pt))==-1) getdomain_error(buff); 
		 /* skipping over 'to' */
		 if(strncmp(&descriptor[pt],"TO",2)!=0) getdomain_error(buff);
		 if((pt=skiptononspace(descriptor,pt))==-1) getdomain_error(buff); 
		 /* cid 2 */
                 if(descriptor[pt]=='_') domain[0].end[nobjects].cid=' ';
                 else domain[0].end[nobjects].cid=descriptor[pt];
                 if((pt=skiptononspace(descriptor,pt))==-1) getdomain_error(buff); 
                 /* n 2 */
                 sscanf(&descriptor[pt],"%d",&domain[0].end[nobjects].n);
                 if((pt=skiptononspace(descriptor,pt))==-1) getdomain_error(buff); 
                 /* ins 2 */
                 if(descriptor[pt]=='_') domain[0].end[nobjects].in=' ';
                 else domain[0].end[nobjects].in=descriptor[pt];
                 pt=skiptononspace(descriptor,pt);
		 nobjects++;
	   } 
	   if(pt!=-1 && descriptor[pt]=='\n') break;
	   /* reallocing if necessary */
	   if(pt!=-1 && strlen(&descriptor[pt])>0 && descriptor[pt]!='\0' && descriptor[pt]!='\n') {
/*		  printf("Allocating memory for a new object !\n"); */
		  domain[0].reverse=(int*)realloc(domain[0].reverse,(nobjects+1)*sizeof(int));
		  domain[0].type=(int*)realloc(domain[0].type,(nobjects+1)*sizeof(int));
	   	  domain[0].start=(struct brookn*)realloc(domain[0].start,(nobjects+1)*sizeof(struct brookn));
		  domain[0].end=(struct brookn*)realloc(domain[0].end,(nobjects+1)*sizeof(struct brookn));
	   }
	      /* now either stop, or move onto the next descriptor */
        } 
	 
	/* check to see whether there is a transformation */
	if(pt!=-1) { /* there is */
	     while(descriptor[pt]!='\n' && descriptor[pt]!='\0') pt++;
	     if(descriptor[pt]=='\0') getdomain_error(buff);
	     (*gottrans)=1;
	     if(sscanf(&descriptor[pt],"%f%f%f%f%f%f%f%f%f%f%f%f",
		&domain[0].R[0][0],&domain[0].R[0][1],&domain[0].R[0][2],&domain[0].V[0],
		&domain[0].R[1][0],&domain[0].R[1][1],&domain[0].R[1][2],&domain[0].V[1],
		&domain[0].R[2][0],&domain[0].R[2][1],&domain[0].R[2][2],&domain[0].V[2])==(char)EOF)
			 getdomain_error(buff);
/*	     printf("Matrix:\n %f %f %f    %f\n %f %f %f   %f\n %f %f %f   %f\n",
		domain[0].R[0][0],domain[0].R[0][1],domain[0].R[0][2],domain[0].V[0],
                domain[0].R[1][0],domain[0].R[1][1],domain[0].R[1][2],domain[0].V[1],
                domain[0].R[2][0],domain[0].R[2][1],domain[0].R[2][2],domain[0].V[2]);  */
	   }
	   domain[0].nobj=nobjects;
	   free(dirfile);
	   free(descriptor);
	   free(buff);
	   free(guff);
	   return 0;
}

int skiptononspace(char *string, int pointer) {
	while(string[pointer]!=' ' && string[pointer]!='\0' && string[pointer]!='\n') pointer++;
	if(string[pointer]=='\0') return -1;
	while(string[pointer]==' ' && string[pointer]!='\0' && string[pointer]!='\n') pointer++;
	if(string[pointer]=='\0') return -1;
	return pointer;
}

void getdomain_error(char *buff) {
	fprintf(stderr,"error in domain descriptors\n");
	fprintf(stderr,"Last domain read:\n%s\n",buff);
	exit(-1);
}
