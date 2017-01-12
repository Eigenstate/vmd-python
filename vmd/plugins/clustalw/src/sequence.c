/********* Sequence input routines for CLUSTAL W *******************/
/* DES was here.  FEB. 1994 */
/* Now reads PILEUP/MSF and CLUSTAL alignment files */

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include "clustalw.h"	

#define MIN(a,b) ((a)<(b)?(a):(b))



/*
*	Prototypes
*/

static char * get_seq(char *,sint *,char *);
static char * get_clustal_seq(char *,sint *,char *,sint);
static char * get_msf_seq(char *,sint *,char *,sint);
static void check_infile(sint *);
static void p_encode(char *, char *, sint);
static void n_encode(char *, char *, sint);
static sint res_index(char *,char);
static Boolean check_dnaflag(char *, sint);
static sint count_clustal_seqs(void);
static sint count_pir_seqs(void);
static sint count_msf_seqs(void);
static sint count_rsf_seqs(void);
static void get_swiss_feature(char *line,sint len);
static void get_rsf_feature(char *line,sint len);
static void get_swiss_mask(char *line,sint len);
static void get_clustal_ss(sint length);
static void get_embl_ss(sint length);
static void get_rsf_ss(sint length);
static void get_gde_ss(sint length);
static Boolean cl_blankline(char *line);

/*
 *	Global variables
 */
extern sint max_names;
FILE *fin;
extern Boolean usemenu, dnaflag, explicit_dnaflag;
extern Boolean interactive;
extern char seqname[];
extern sint nseqs;
extern sint *seqlen_array;
extern sint *output_index;
extern char **names,**titles;
extern char **seq_array;
extern Boolean profile1_empty, profile2_empty;
extern sint gap_pos2;
extern sint max_aln_length;
extern char *gap_penalty_mask, *sec_struct_mask;
extern sint struct_penalties;
extern char *ss_name;
extern sint profile_no;
extern sint debug;

char *amino_acid_codes   =    "ABCDEFGHIKLMNPQRSTUVWXYZ-";  /* DES */
static sint seqFormat;
static char chartab[128];
static char *formatNames[] = {"unknown","EMBL/Swiss-Prot","PIR",
			      "Pearson","GDE","Clustal","Pileup/MSF","RSF","USER","PHYLIP","NEXUS"};

void fill_chartab(void)	/* Create translation and check table */
{
	register sint i;
	register char c;
	
	for(i=0;i<128;chartab[i++]=0);
	for(i=0;(c=amino_acid_codes[i]);i++)
		chartab[(int)c]=chartab[tolower(c)]=c;
}

static char * get_msf_seq(char *sname,sint *len,char *tit,sint seqno)
/* read the seqno_th. sequence from a PILEUP multiple alignment file */
{
	static char line[MAXLINE+1];
	char *seq = NULL;
	sint i,j,k;
	unsigned char c;

	fseek(fin,0,0); 		/* start at the beginning */

	*len=0;				/* initialise length to zero */
        for(i=0;;i++) {
		if(fgets(line,MAXLINE+1,fin)==NULL) return NULL; /* read the title*/
		if(linetype(line,"//") ) break;		    /* lines...ignore*/
	}

	while (fgets(line,MAXLINE+1,fin) != NULL) {
		if(!blankline(line)) {

			for(i=1;i<seqno;i++) fgets(line,MAXLINE+1,fin);
                        for(j=0;j<=strlen(line);j++) if(line[j] != ' ') break;
			for(k=j;k<=strlen(line);k++) if(line[k] == ' ') break;
			strncpy(sname,line+j,MIN(MAXNAMES,k-j)); 
			sname[MIN(MAXNAMES,k-j)]=EOS;
			rtrim(sname);
                       	blank_to_(sname);

			if(seq==NULL)
				seq=(char *)ckalloc((MAXLINE+2)*sizeof(char));
			else
				seq=(char *)ckrealloc(seq,((*len)+MAXLINE+2)*sizeof(char));
			for(i=k;i<=MAXLINE;i++) {
				c=line[i];
				if(c == '.' || c == '~' ) c = '-';
				if(c == '*') c = 'X';
				if(c == '\n' || c == EOS) break; /* EOL */
				c=chartab[c];
				if(c) seq[++(*len)]=c;
			}

			for(i=0;;i++) {
				if(fgets(line,MAXLINE+1,fin)==NULL) return seq;
				if(blankline(line)) break;
			}
		}
	}
	return seq;
}

static Boolean cl_blankline(char *line)
{
	int i;

	if (line[0] == '!') return TRUE;
	
	for(i=0;line[i]!='\n' && line[i]!=EOS;i++) {
		if( isdigit(line[i]) ||
		    isspace(line[i]) ||
		    (line[i] == '*') ||
		    (line[i] == ':') ||
                    (line[i] == '.')) 
			;
		else
			return FALSE;
	}
	return TRUE;
}

static char * get_clustal_seq(char *sname,sint *len,char *tit,sint seqno)
/* read the seqno_th. sequence from a clustal multiple alignment file */
{
	static char line[MAXLINE+1];
	static char tseq[MAXLINE+1];
	char *seq = NULL;
	sint i,j;
	unsigned char c;

	fseek(fin,0,0); 		/* start at the beginning */

	*len=0;				/* initialise length to zero */
	fgets(line,MAXLINE+1,fin);	/* read the title line...ignore it */

	while (fgets(line,MAXLINE+1,fin) != NULL) {
		if(!cl_blankline(line)) {

			for(i=1;i<seqno;i++) fgets(line,MAXLINE+1,fin);
			for(j=0;j<=strlen(line);j++) if(line[j] != ' ') break;

			sscanf(line,"%s%s",sname,tseq);
			for(j=0;j<MAXNAMES;j++) if(sname[j] == ' ') break;
			sname[j]=EOS;
			rtrim(sname);
                       	blank_to_(sname);

			if(seq==NULL)
				seq=(char *)ckalloc((MAXLINE+2)*sizeof(char));
			else
				seq=(char *)ckrealloc(seq,((*len)+MAXLINE+2)*sizeof(char));
			for(i=0;i<=MAXLINE;i++) {
				c=tseq[i];
				/*if(c == '\n' || c == EOS) break;*/ /* EOL */
				if(isspace(c) || c == EOS) break; /* EOL */
				c=chartab[c];
				if(c) seq[++(*len)]=c;
			}

			for(i=0;;i++) {
				if(fgets(line,MAXLINE+1,fin)==NULL) return seq;
				if(cl_blankline(line)) break;
			}
		}
	}

	return seq;
}

static void get_clustal_ss(sint length)
/* read the structure data from a clustal multiple alignment file */
{
	static char title[MAXLINE+1];
	static char line[MAXLINE+1];
	static char lin2[MAXLINE+1];
	static char tseq[MAXLINE+1];
	static char sname[MAXNAMES+1];
	sint i,j,len,ix,struct_index=0;
	char c;

	
	fseek(fin,0,0); 		/* start at the beginning */

	len=0;				/* initialise length to zero */
	if (fgets(line,MAXLINE+1,fin) == NULL) return;	/* read the title line...ignore it */

	if (fgets(line,MAXLINE+1,fin) == NULL) return;  /* read the next line... */
/* skip any blank lines */
	for (;;) {
		if(fgets(line,MAXLINE+1,fin)==NULL) return;
		if(!blankline(line)) break;
	}

/* look for structure table lines */
	ix = -1;
	for(;;) {
		if(line[0] != '!') break;
		if(strncmp(line,"!SS",3) == 0) {
			ix++;
			sscanf(line+4,"%s%s",sname,tseq);
			for(j=0;j<MAXNAMES;j++) if(sname[j] == ' ') break;
			sname[j]=EOS;
			rtrim(sname);
    		blank_to_(sname);
    		if (interactive) {
				strcpy(title,"Found secondary structure in alignment file: ");
				strcat(title,sname);
				(*lin2)=prompt_for_yes_no(title,"Use it to set local gap penalties ");
			}
			else (*lin2) = 'y';
			if ((*lin2 != 'n') && (*lin2 != 'N'))  {               	
				struct_penalties = SECST;
				struct_index = ix;
				for (i=0;i<length;i++)
				{
					sec_struct_mask[i] = '.';
					gap_penalty_mask[i] = '.';
				}
				strcpy(ss_name,sname);
				for(i=0;len < length;i++) {
					c = tseq[i];
					if(c == '\n' || c == EOS) break; /* EOL */
					if (!isspace(c)) sec_struct_mask[len++] = c;
				}
			}
		}
		else if(strncmp(line,"!GM",3) == 0) {
			ix++;
			sscanf(line+4,"%s%s",sname,tseq);
			for(j=0;j<MAXNAMES;j++) if(sname[j] == ' ') break;
			sname[j]=EOS;
			rtrim(sname);
    		blank_to_(sname);
    		if (interactive) {
				strcpy(title,"Found gap penalty mask in alignment file: ");
				strcat(title,sname);
				(*lin2)=prompt_for_yes_no(title,"Use it to set local gap penalties ");
			}
			else (*lin2) = 'y';
			if ((*lin2 != 'n') && (*lin2 != 'N'))  {               	
				struct_penalties = GMASK;
				struct_index = ix;
				for (i=0;i<length;i++)
					gap_penalty_mask[i] = '1';
					strcpy(ss_name,sname);
				for(i=0;len < length;i++) {
					c = tseq[i];
					if(c == '\n' || c == EOS) break; /* EOL */
					if (!isspace(c)) gap_penalty_mask[len++] = c;
				}
			}
		}
		if (struct_penalties != NONE) break;
		if(fgets(line,MAXLINE+1,fin)==NULL) return;
	}
			
	if (struct_penalties == NONE) return;
	
/* skip any more comment lines */
	while (line[0] == '!') {
		if(fgets(line,MAXLINE+1,fin)==NULL) return;
	}

/* skip the sequence lines and any comments after the alignment */
	for (;;) {
		if(isspace(line[0])) break;
		if(fgets(line,MAXLINE+1,fin)==NULL) return;
	}
			

/* read the rest of the alignment */
	
	for (;;) {
/* skip any blank lines */
			for (;;) {
				if(!blankline(line)) break;
				if(fgets(line,MAXLINE+1,fin)==NULL) return;
			}
/* get structure table line */
			for(ix=0;ix<struct_index;ix++) {
				if (line[0] != '!') {
					if(struct_penalties == SECST)
						error("bad secondary structure format");
					else
						error("bad gap penalty mask format");
				   	struct_penalties = NONE;
					return;
				}
				if(fgets(line,MAXLINE+1,fin)==NULL) return;
			}
			if(struct_penalties == SECST) {
				if (strncmp(line,"!SS",3) != 0) {
					error("bad secondary structure format");
					struct_penalties = NONE;
					return;
				}
				sscanf(line+4,"%s%s",sname,tseq);
				for(i=0;len < length;i++) {
					c = tseq[i];
					if(c == '\n' || c == EOS) break; /* EOL */
					if (!isspace(c)) sec_struct_mask[len++] = c;
				}			
			}
			else if (struct_penalties == GMASK) {
				if (strncmp(line,"!GM",3) != 0) {
					error("bad gap penalty mask format");
					struct_penalties = NONE;
					return;
				}
				sscanf(line+4,"%s%s",sname,tseq);
				for(i=0;len < length;i++) {
					c = tseq[i];
					if(c == '\n' || c == EOS) break; /* EOL */
					if (!isspace(c)) gap_penalty_mask[len++] = c;
				}			
			}

/* skip any more comment lines */
		while (line[0] == '!') {
			if(fgets(line,MAXLINE+1,fin)==NULL) return;
		}

/* skip the sequence lines */
		for (;;) {
			if(isspace(line[0])) break;
			if(fgets(line,MAXLINE+1,fin)==NULL) return;
		}
	}
}

static void get_embl_ss(sint length)
{
	static char title[MAXLINE+1];
	static char line[MAXLINE+1];
	static char lin2[MAXLINE+1];
	static char sname[MAXNAMES+1];
	char feature[MAXLINE+1];
	sint i;

/* find the start of the sequence entry */
	for (;;) {
		while( !linetype(line,"ID") )
			if (fgets(line,MAXLINE+1,fin) == NULL) return;
			
    	for(i=5;i<=strlen(line);i++)  /* DES */
			if(line[i] != ' ') break;
		strncpy(sname,line+i,MAXNAMES); /* remember entryname */
    		for(i=0;i<=strlen(sname);i++)
			if(sname[i] == ' ') {
				sname[i]=EOS;
				break;
			}
		sname[MAXNAMES]=EOS;
		rtrim(sname);
    	blank_to_(sname);
		
/* look for secondary structure feature table / gap penalty mask */
		while(fgets(line,MAXLINE+1,fin) != NULL) {
			if (linetype(line,"FT")) {
				sscanf(line+2,"%s",feature);
				if (strcmp(feature,"HELIX") == 0 ||
				    strcmp(feature,"STRAND") == 0)
				{

				if (interactive) {
					strcpy(title,"Found secondary structure in alignment file: ");
					strcat(title,sname);
					(*lin2)=prompt_for_yes_no(title,"Use it to set local gap penalties ");
				}
				else (*lin2) = 'y';
				if ((*lin2 != 'n') && (*lin2 != 'N'))  {               	
					struct_penalties = SECST;
					for (i=0;i<length;i++)
						sec_struct_mask[i] = '.';
					do {
						get_swiss_feature(&line[2],length);
						fgets(line,MAXLINE+1,fin);
					} while( linetype(line,"FT") );
				}
				else {
					do {
						fgets(line,MAXLINE+1,fin);
					} while( linetype(line,"FT") );
				}
				strcpy(ss_name,sname);
				}
			}
			else if (linetype(line,"GM")) {
				if (interactive) {
					strcpy(title,"Found gap penalty mask in alignment file: ");
					strcat(title,sname);
					(*lin2)=prompt_for_yes_no(title,"Use it to set local gap penalties ");
				}
				else (*lin2) = 'y';
				if ((*lin2 != 'n') && (*lin2 != 'N'))  {               	
					struct_penalties = GMASK;
					for (i=0;i<length;i++)
						gap_penalty_mask[i] = '1';
					do {
						get_swiss_mask(&line[2],length);
						fgets(line,MAXLINE+1,fin);
					} while( linetype(line,"GM") );
				}
				else {
					do {
						fgets(line,MAXLINE+1,fin);
					} while( linetype(line,"GM") );
				}
				strcpy(ss_name,sname);
			}
			if (linetype(line,"SQ"))
				break;	

			if (struct_penalties != NONE) break;			
		}
						
	}
						
}

static void get_rsf_ss(sint length)
{
	static char title[MAXLINE+1];
	static char line[MAXLINE+1];
	static char lin2[MAXLINE+1];
	static char sname[MAXNAMES+1];
	sint i;

/* skip the comments */
	while (fgets(line,MAXLINE+1,fin) != NULL) {
 		if(line[strlen(line)-2]=='.' &&
                                 line[strlen(line)-3]=='.')
			break;
	}

/* find the start of the sequence entry */
	for (;;) {
		while (fgets(line,MAXLINE+1,fin) != NULL)
                	if( *line == '{' ) break;

		while( !keyword(line,"name") )
			if (fgets(line,MAXLINE+1,fin) == NULL) return;
			
    	for(i=5;i<=strlen(line);i++)  /* DES */
			if(line[i] != ' ') break;
		strncpy(sname,line+i,MAXNAMES); /* remember entryname */
    		for(i=0;i<=strlen(sname);i++)
			if(sname[i] == ' ') {
				sname[i]=EOS;
				break;
			}
		sname[MAXNAMES]=EOS;
		rtrim(sname);
    	blank_to_(sname);
		
/* look for secondary structure feature table / gap penalty mask */
		while(fgets(line,MAXLINE+1,fin) != NULL) {
			if (keyword(line,"feature")) {
				if (interactive) {
					strcpy(title,"Found secondary structure in alignment file: ");
					strcat(title,sname);
					(*lin2)=prompt_for_yes_no(title,"Use it to set local gap penalties ");
				}
				else (*lin2) = 'y';
				if ((*lin2 != 'n') && (*lin2 != 'N'))  {               	
					struct_penalties = SECST;
					for (i=0;i<length;i++)
						sec_struct_mask[i] = '.';
					do {
						if(keyword(line,"feature"))
							get_rsf_feature(&line[7],length);
						fgets(line,MAXLINE+1,fin);
					} while( !keyword(line,"sequence") );
				}
				else {
					do {
						fgets(line,MAXLINE+1,fin);
					} while( !keyword(line,"sequence") );
				}
				strcpy(ss_name,sname);
			}
			else if (keyword(line,"sequence"))
				break;	

			if (struct_penalties != NONE) break;			
		}
						
	}
						
}

static void get_gde_ss(sint length)
{
	static char title[MAXLINE+1];
	static char line[MAXLINE+1];
	static char lin2[MAXLINE+1];
	static char sname[MAXNAMES+1];
	sint i, len, offset = 0;
        unsigned char c;

	for (;;) {
		line[0] = '\0';
/* search for the next comment line */
		while(*line != '"')
			if (fgets(line,MAXLINE+1,fin) == NULL) return;

/* is it a secondary structure entry? */
		if (strncmp(&line[1],"SS_",3) == 0) {
			for (i=1;i<=MAXNAMES-3;i++) {
				if (line[i+3] == '(' || line[i+3] == '\n')
						break;
				sname[i-1] = line[i+3];
			}
			i--;
			sname[i]=EOS;
			if (sname[i-1] == '(') sscanf(&line[i+3],"%d",&offset);
			else offset = 0;
			for(i--;i > 0;i--) 
				if(isspace(sname[i])) {
					sname[i]=EOS;	
				}
				else break;		
			blank_to_(sname);

			if (interactive) {
				strcpy(title,"Found secondary structure in alignment file: ");
				strcat(title,sname);
				(*lin2)=prompt_for_yes_no(title,"Use it to set local gap penalties ");
			}
			else (*lin2) = 'y';
			if ((*lin2 != 'n') && (*lin2 != 'N'))  {               	
				struct_penalties = SECST;
				for (i=0;i<length;i++)
					sec_struct_mask[i] = '.';
				len = 0;
				while(fgets(line,MAXLINE+1,fin)) {
					if(*line == '%' || *line == '#' || *line == '"') break;
					for(i=offset;i < length;i++) {
						c=line[i];
						if(c == '\n' || c == EOS) 
							break;			/* EOL */
						sec_struct_mask[len++]=c;
					}
					if (len > length) break;
				}
				strcpy(ss_name,sname);
			}
		}
/* or is it a gap penalty mask entry? */
		else if (strncmp(&line[1],"GM_",3) == 0) {
			for (i=1;i<=MAXNAMES-3;i++) {
				if (line[i+3] == '(' || line[i+3] == '\n')
						break;
				sname[i-1] = line[i+3];
			}
			i--;
			sname[i]=EOS;
			if (sname[i-1] == '(') sscanf(&line[i+3],"%d",&offset);
			else offset = 0;
			for(i--;i > 0;i--) 
				if(isspace(sname[i])) {
					sname[i]=EOS;	
				}
				else break;		
			blank_to_(sname);

			if (interactive) {
				strcpy(title,"Found gap penalty mask in alignment file: ");
				strcat(title,sname);
				(*lin2)=prompt_for_yes_no(title,"Use it to set local gap penalties ");
			}
			else (*lin2) = 'y';
			if ((*lin2 != 'n') && (*lin2 != 'N'))  {               	
				struct_penalties = GMASK;
				for (i=0;i<length;i++)
					gap_penalty_mask[i] = '1';
				len = 0;
				while(fgets(line,MAXLINE+1,fin)) {
					if(*line == '%' || *line == '#' || *line == '"') break;
					for(i=offset;i < length;i++) {
						c=line[i];
						if(c == '\n' || c == EOS) 
							break;			/* EOL */
						gap_penalty_mask[len++]=c;
					}
					if (len > length) break;
				}
				strcpy(ss_name,sname);
			}
		}
		if (struct_penalties != NONE) break;			
	}			
			
}

static void get_swiss_feature(char *line, sint len)
{
	char c, s, feature[MAXLINE+1];
	int  i, start_pos, end_pos;
	
	if (sscanf(line,"%s%d%d",feature,&start_pos,&end_pos) != 3) {
		return;
	}

	if (strcmp(feature,"HELIX") == 0) {
		c = 'A';
		s = '$';
	}
	else if (strcmp(feature,"STRAND") == 0) {
		c = 'B';
		s = '%';
	}
	else
		return;
			
	if(start_pos >=len || end_pos>=len) return;

	sec_struct_mask[start_pos-1] = s;
	for (i=start_pos;i<end_pos-1;i++)
		sec_struct_mask[i] = c;
	sec_struct_mask[end_pos-1] = s;
		
}

static void get_rsf_feature(char *line, sint len)
{
	char c, s;
	char str1[MAXLINE+1],str2[MAXLINE+1],feature[MAXLINE+1];
	int  i, tmp,start_pos, end_pos;
	
	if (sscanf(line,"%d%d%d%s%s%s",&start_pos,&end_pos,&tmp,str1,str2,feature) != 6) {
		return;
	}

	if (strcmp(feature,"HELIX") == 0) {
		c = 'A';
		s = '$';
	}
	else if (strcmp(feature,"STRAND") == 0) {
		c = 'B';
		s = '%';
	}
	else
		return;
			
	if(start_pos>=len || end_pos >= len) return;
	sec_struct_mask[start_pos-1] = s;
	for (i=start_pos;i<end_pos-1;i++)
		sec_struct_mask[i] = c;
	sec_struct_mask[end_pos-1] = s;
		
}

static void get_swiss_mask(char *line, sint len)
{
	int  i, value, start_pos, end_pos;
	
	if (sscanf(line,"%d%d%d",&value,&start_pos,&end_pos) != 3) {
		return;
	}

	if (value < 1 || value > 9) return;
	
	if(start_pos>=len || end_pos >= len) return;
	for (i=start_pos-1;i<end_pos;i++)
		gap_penalty_mask[i] = value+'0';
		
}

static char * get_seq(char *sname,sint *len,char *tit)
{
	static char line[MAXLINE+1];
	char *seq = NULL;
	sint i, offset = 0;
        unsigned char c=EOS;
	Boolean got_seq=FALSE;

	switch(seqFormat) {

/************************************/
		case EMBLSWISS:
			while( !linetype(line,"ID") )
				if (fgets(line,MAXLINE+1,fin) == NULL) return NULL;
			
                        for(i=5;i<=strlen(line);i++)  /* DES */
				if(line[i] != ' ') break;
			strncpy(sname,line+i,MAXNAMES); /* remember entryname */
                	for(i=0;i<=strlen(sname);i++)
                        	if(sname[i] == ' ') {
                                	sname[i]=EOS;
                                	break;
                        	}

			sname[MAXNAMES]=EOS;
			rtrim(sname);
                        blank_to_(sname);

						
			while( !linetype(line,"SQ") )
				fgets(line,MAXLINE+1,fin);
				
			*len=0;
			while(fgets(line,MAXLINE+1,fin)) {
				if(got_seq && blankline(line)) break;
 				if( strlen(line) > 2 && line[strlen(line)-2]=='.' && line[strlen(line)-3]=='.' ) 
					continue;
				if(seq==NULL)
					seq=(char *)ckalloc((MAXLINE+2)*sizeof(char));
				else
					seq=(char *)ckrealloc(seq,((*len)+MAXLINE+2)*sizeof(char));
				for(i=0;i<=MAXLINE;i++) {
					c=line[i];
				if(c == '\n' || c == EOS || c == '/')
					break;			/* EOL */
				c=chartab[c];
				if(c) {
					got_seq=TRUE;
					seq[++(*len)]=c;
				}
				}
			if(c == '/') break;
			}
		break;
		
/************************************/
		case PIR:
			while(*line != '>')
				fgets(line,MAXLINE+1,fin);			
                        for(i=4;i<=strlen(line);i++)  /* DES */
				if(line[i] != ' ') break;
			strncpy(sname,line+i,MAXNAMES); /* remember entryname */
			sname[MAXNAMES]=EOS;
			rtrim(sname);
                        blank_to_(sname);

			fgets(line,MAXLINE+1,fin);
			strncpy(tit,line,MAXTITLES);
			tit[MAXTITLES]=EOS;
			i=strlen(tit);
			if(tit[i-1]=='\n') tit[i-1]=EOS;
			
			*len=0;
			while(fgets(line,MAXLINE+1,fin)) {
				if(seq==NULL)
					seq=(char *)ckalloc((MAXLINE+2)*sizeof(char));
				else
					seq=(char *)ckrealloc(seq,((*len)+MAXLINE+2)*sizeof(char));
				for(i=0;i<=MAXLINE;i++) {
					c=line[i];
				if(c == '\n' || c == EOS || c == '*')
					break;			/* EOL */
			
				c=chartab[c];
				if(c) seq[++(*len)]=c;
				}
			if(c == '*') break;
			}
		break;
/***********************************************/
		case PEARSON:
			while(*line != '>')
				fgets(line,MAXLINE+1,fin);
			
                        for(i=1;i<=strlen(line);i++)  /* DES */
				if(line[i] != ' ') break;
			strncpy(sname,line+i,MAXNAMES); /* remember entryname */
                        for(i=1;i<=strlen(sname);i++)  /* DES */
				if(sname[i] == ' ') break;
			sname[i]=EOS;
			rtrim(sname);
                        blank_to_(sname);

			*tit=EOS;
			
			*len=0;
			while(fgets(line,MAXLINE+1,fin)) {
				if(seq==NULL)
					seq=(char *)ckalloc((MAXLINE+2)*sizeof(char));
				else
					seq=(char *)ckrealloc(seq,((*len)+MAXLINE+2)*sizeof(char));
				for(i=0;i<=MAXLINE;i++) {
					c=line[i];
				if(c == '\n' || c == EOS || c == '>')
					break;			/* EOL */
			
				c=chartab[c];
				if(c) seq[++(*len)]=c;
			}
			if(c == '>') break;
			}
		break;
/**********************************************/
		case GDE:
			if (dnaflag) {
				while(*line != '#')
					fgets(line,MAXLINE+1,fin);
			}
			else {
				while(*line != '%')
					fgets(line,MAXLINE+1,fin);
			}
			
			for (i=1;i<=MAXNAMES;i++) {
				if (line[i] == '(' || line[i] == '\n')
                                    break;
				sname[i-1] = line[i];
			}
			i--;
			sname[i]=EOS;
			if (sname[i-1] == '(') sscanf(&line[i],"%d",&offset);
			else offset = 0;
			for(i--;i > 0;i--) 
				if(isspace(sname[i])) {
					sname[i]=EOS;	
				}
				else break;		
                        blank_to_(sname);

			*tit=EOS;
			
			*len=0;
			for (i=0;i<offset;i++) seq[++(*len)] = '-';
			while(fgets(line,MAXLINE+1,fin)) {
			if(*line == '%' || *line == '#' || *line == '"') break;
				if(seq==NULL)
					seq=(char *)ckalloc((MAXLINE+2)*sizeof(char));
				else
					seq=(char *)ckrealloc(seq,((*len)+MAXLINE+2)*sizeof(char));
				for(i=0;i<=MAXLINE;i++) {
					c=line[i];
				if(c == '\n' || c == EOS) 
					break;			/* EOL */
			
				c=chartab[c];
				if(c) seq[++(*len)]=c;
				}
			}
		break;
/***********************************************/
		case RSF:
			while(*line != '{')
				if (fgets(line,MAXLINE+1,fin) == NULL) return NULL;
			
			while( !keyword(line,"name") )
				if (fgets(line,MAXLINE+1,fin) == NULL) return NULL;
			
                        for(i=5;i<=strlen(line);i++)  /* DES */
				if(line[i] != ' ') break;
			strncpy(sname,line+i,MAXNAMES); /* remember entryname */
                	for(i=0;i<=strlen(sname);i++)
                        	if(sname[i] == ' ') {
                                	sname[i]=EOS;
                                	break;
                        	}

			sname[MAXNAMES]=EOS;
			rtrim(sname);
                        blank_to_(sname);

						
			while( !keyword(line,"sequence") )
				if (fgets(line,MAXLINE+1,fin) == NULL) return NULL;
				
			*len=0;
			while(fgets(line,MAXLINE+1,fin)) {
				if(seq==NULL)
					seq=(char *)ckalloc((MAXLINE+2)*sizeof(char));
				else
					seq=(char *)ckrealloc(seq,((*len)+MAXLINE+2)*sizeof(char));
				for(i=0;i<=MAXLINE;i++) {
					c=line[i];
					if(c == EOS || c == '}')
						break;			/* EOL */
					if( c=='.')
						seq[++(*len)]='-';
					c=chartab[c];
					if(c) seq[++(*len)]=c;
				}
				if(c == '}') break;
			}
		break;
/***********************************************/
	}
	
	seq[*len+1]=EOS;

	return seq;
}


sint readseqs(sint first_seq) /*first_seq is the #no. of the first seq. to read */
{
	char line[FILENAMELEN+1];
	char fileName[FILENAMELEN+1];

	static char *seq1,sname1[MAXNAMES+1],title[MAXTITLES+1];
	sint i,j;
	sint no_seqs;
	static sint l1;
	static Boolean dnaflag1;
	
	if(usemenu)
		getstr("Enter the name of the sequence file",line, FILENAMELEN);
	else
		strcpy(line,seqname);
	if(*line == EOS) return -1;

	if ((sscanf(line,"file://%s",fileName) == 1 )) {
	  strcpy(line,fileName);
	}

	if((fin=fopen(line,"r"))==NULL) {
		error("Could not open sequence file (%s) ",line);
		return -1;      /* DES -1 => file not found */
	}
	strcpy(seqname,line);
	no_seqs=0;
	check_infile(&no_seqs);
	info("Sequence format is %s",formatNames[seqFormat]);
	if(seqFormat==NEXUS)
		error("Cannot read nexus format");

/* DES DEBUG 
	fprintf(stdout,"\n\n File name = %s\n\n",seqname);
*/
	if(no_seqs == 0)
		return 0;       /* return the number of seqs. (zero here)*/

/*
	if((no_seqs + first_seq -1) > MAXN) {
		error("Too many sequences. Maximum is %d",(pint)MAXN);
		return 0;
	}
*/

/* DES */
/*	if(seqFormat == CLUSTAL) {
		info("no of sequences = %d",(pint)no_seqs);
		return no_seqs;
	}
*/
	max_aln_length = 0;

/* if this is a multiple alignment, or profile 1 - free any memory used
by previous alignments, then allocate memory for the new alignment */
	if(first_seq == 1) {
		max_names = 0;
		free_aln(nseqs);
		alloc_aln(no_seqs);
	}
/* otherwise, this is a profile 2, and we need to reallocate the arrays,
leaving the data for profile 1 intact */
	else realloc_aln(first_seq,no_seqs);

        for(i=1;i<first_seq;i++)
	{
                if(seqlen_array[i]>max_aln_length)
                        max_aln_length=seqlen_array[i];
		if(strlen(names[i])>max_names)
			max_names=strlen(names[i]);
	}

	for(i=first_seq;i<=first_seq+no_seqs-1;i++) {    /* get the seqs now*/
		output_index[i] = i;	/* default output order */
		if(seqFormat == CLUSTAL) 
			seq1=get_clustal_seq(sname1,&l1,title,i-first_seq+1);
		else if(seqFormat == MSF)
			    seq1=get_msf_seq(sname1,&l1,title,i-first_seq+1);
		else
			seq1=get_seq(sname1,&l1,title);

		if(seq1==NULL) break;
/* JULIE */
/*  Set max length of dynamically allocated arrays in prfalign.c */
		if (l1 > max_aln_length) max_aln_length = l1;
		seqlen_array[i]=l1;                   /* store the length */
		strcpy(names[i],sname1);              /*    "   "  name   */
		strcpy(titles[i],title);              /*    "   "  title  */

		if(!explicit_dnaflag) {
			dnaflag1 = check_dnaflag(seq1,l1); /* check DNA/Prot */
		        if(i == 1) dnaflag = dnaflag1;
		}			/* type decided by first seq*/
		else
			dnaflag1 = dnaflag;

		alloc_seq(i,l1);

		if(dnaflag)
			n_encode(seq1,seq_array[i],l1); /* encode the sequence*/
		else					/* as ints  */
			p_encode(seq1,seq_array[i],l1);
		if(seq1!=NULL) seq1=ckfree(seq1);
	}


	max_aln_length *= 2;
/*
   JULIE
   check sequence names are all different - otherwise phylip tree is 
   confused.
*/
	for(i=1;i<=first_seq+no_seqs-1;i++) {
		for(j=i+1;j<=first_seq+no_seqs-1;j++) {
			if (strncmp(names[i],names[j],MAXNAMES) == 0) {
				error("Multiple sequences found with same name, %s (first %d chars are significant)", names[i],MAXNAMES);
				return 0;
			}
		}
	}
	for(i=first_seq;i<=first_seq+no_seqs-1;i++)
	{
		if(seqlen_array[i]>max_aln_length)
			max_aln_length=seqlen_array[i];
	}
	
/* look for a feature table / gap penalty mask (only if this is a profile) */
	if (profile_no > 0) {
		rewind(fin);
		struct_penalties = NONE;
    		gap_penalty_mask = (char *)ckalloc((max_aln_length+1) * sizeof (char));
    		sec_struct_mask = (char *)ckalloc((max_aln_length+1) * sizeof (char));
    		ss_name = (char *)ckalloc((MAXNAMES+1) * sizeof (char));

		if (seqFormat == CLUSTAL) {
			get_clustal_ss(max_aln_length);
		}
		else if (seqFormat == GDE) {
			get_gde_ss(max_aln_length);
		}
		else if (seqFormat == EMBLSWISS) {
			get_embl_ss(max_aln_length);
		}
		else if (seqFormat == RSF) {
			get_rsf_ss(max_aln_length);
		}
	}

	for(i=first_seq;i<=first_seq+no_seqs-1;i++)
	{
		if(strlen(names[i])>max_names)
			max_names=strlen(names[i]);
	}

	if(max_names<10) max_names=10;

	fclose(fin);
			
	return no_seqs;    /* return the number of seqs. read in this call */
}


static Boolean check_dnaflag(char *seq, sint slen)
/* check if DNA or Protein
   The decision is based on counting all A,C,G,T,U or N. 
   If >= 85% of all characters (except -) are as above => DNA  */
{
	sint i, c, nresidues, nbases;
	float ratio;
	char *dna_codes="ACGTUN";
	
	nresidues = nbases = 0;	
	for(i=1; i <= slen; i++) {
		if(seq[i] != '-') {
			nresidues++;
			if(seq[i] == 'N')
				nbases++;
			else {
				c = res_index(dna_codes, seq[i]);
				if(c >= 0)
					nbases++;
			}
		}
	}
	if( (nbases == 0) || (nresidues == 0) ) return FALSE;
	ratio = (float)nbases/(float)nresidues;
/* DES 	fprintf(stdout,"\n nbases = %d, nresidues = %d, ratio = %f\n",
		(pint)nbases,(pint)nresidues,(pint)ratio); */
	if(ratio >= 0.85) 
		return TRUE;
	else
		return FALSE;
}



static void check_infile(sint *nseqs)
{
	char line[MAXLINE+1];
	sint i;	

	*nseqs=0;
	while (fgets(line,MAXLINE+1,fin) != NULL) {
		if(!blankline(line)) 
			break;
	}

	for(i=strlen(line)-1;i>=0;i--)
		if(isgraph(line[i])) break;
	line[i+1]=EOS;
        
	for(i=0;i<=6;i++) line[i] = toupper(line[i]);

	if( linetype(line,"ID") ) {					/* EMBL/Swiss-Prot format ? */
		seqFormat=EMBLSWISS;
		(*nseqs)++;
	}
        else if( linetype(line,"CLUSTAL") ) {
		seqFormat=CLUSTAL;
	}
 	else if( linetype(line,"PILEUP") ) {
		seqFormat = MSF;
	}
 	else if( linetype(line,"!!AA_MULTIPLE_ALIGNMENT") ) {
		seqFormat = MSF;
		dnaflag = FALSE;
	}
 	else if( linetype(line,"!!NA_MULTIPLE_ALIGNMENT") ) {
		seqFormat = MSF;
		dnaflag = TRUE;
	}
 	else if( strstr(line,"MSF") && line[strlen(line)-1]=='.' &&
                                 line[strlen(line)-2]=='.' ) {
		seqFormat = MSF;
	}
 	else if( linetype(line,"!!RICH_SEQUENCE") ) {
		seqFormat = RSF;
	}
 	else if( linetype(line,"#NEXUS") ) {
		seqFormat=NEXUS;
		return;
	}
	else if(*line == '>') {						/* no */
		seqFormat=(line[3] == ';')?PIR:PEARSON; /* distinguish PIR and Pearson */
		(*nseqs)++;
	}
	else if((*line == '"') || (*line == '%') || (*line == '#')) {
		seqFormat=GDE; /* GDE format */
		if (*line == '%') {
                        (*nseqs)++;
			dnaflag = FALSE;
		}
		else if (*line == '#') {
			(*nseqs)++;
			dnaflag = TRUE;
		}
	}
	else {
		seqFormat=UNKNOWN;
		return;
	}

	while(fgets(line,MAXLINE+1,fin) != NULL) {
		switch(seqFormat) {
			case EMBLSWISS:
				if( linetype(line,"ID") )
					(*nseqs)++;
				break;
			case PIR:
				*nseqs = count_pir_seqs();
				fseek(fin,0,0);
				return;
			case PEARSON:
                                if( *line == '>' )
                                        (*nseqs)++;
                                break;
			case GDE:
				if(( *line == '%' ) && ( dnaflag == FALSE))
					(*nseqs)++;
				else if (( *line == '#') && ( dnaflag == TRUE))
					(*nseqs)++;
				break;
			case CLUSTAL:
				*nseqs = count_clustal_seqs();
/* DES */ 			/* fprintf(stdout,"\nnseqs = %d\n",(pint)*nseqs); */
				fseek(fin,0,0);
				return;
			case MSF:
				*nseqs = count_msf_seqs();
				fseek(fin,0,0);
				return;
			case RSF:
				fseek(fin,0,0);
				*nseqs = count_rsf_seqs();
				fseek(fin,0,0);
				return;
			case USER:
			default:
				break;
		}
	}
	fseek(fin,0,0);
}


static sint count_pir_seqs(void)
/* count the number of sequences in a pir alignment file */
{
	char line[MAXLINE+1],c;
	sint  nseqs, i;
	Boolean seq_ok;

	seq_ok = FALSE;
	while (fgets(line,MAXLINE+1,fin) != NULL) { /* Look for end of first seq */
		if(*line == '>') break;
		for(i=0;seq_ok == FALSE;i++) {
			c=line[i];
			if(c == '*') {
				seq_ok = TRUE;	/* ok - end of sequence found */
				break;
			}			/* EOL */
			if(c == '\n' || c == EOS)
				break;			/* EOL */
		}
		if (seq_ok == TRUE)
			break;
	}
	if (seq_ok == FALSE) {
		error("PIR format sequence end marker '*'\nmissing for one or more sequences.");
		return (sint)0;	/* funny format*/
	}
	
	
	nseqs = 1;
	
	while (fgets(line,MAXLINE+1,fin) != NULL) {
		if(*line == '>') {		/* Look for start of next seq */
			seq_ok = FALSE;
			while (fgets(line,MAXLINE+1,fin) != NULL) { /* Look for end of seq */
				if(*line == '>') {
					error("PIR format sequence end marker '*' missing for one or more sequences.");
					return (sint)0;	/* funny format*/
				}
				for(i=0;seq_ok == FALSE;i++) {
					c=line[i];
					if(c == '*') {
						seq_ok = TRUE;	/* ok - sequence found */
						break;
					}			/* EOL */
					if(c == '\n' || c == EOS)
						break;			/* EOL */
				}
				if (seq_ok == TRUE) {
					nseqs++;
					break;
				}
			}
		}
	}
	return (sint)nseqs;
}


static sint count_clustal_seqs(void)
/* count the number of sequences in a clustal alignment file */
{
	char line[MAXLINE+1];
	sint  nseqs;

	while (fgets(line,MAXLINE+1,fin) != NULL) {
		if(!cl_blankline(line)) break;		/* Look for next non- */
	}						/* blank line */
	nseqs = 1;

	while (fgets(line,MAXLINE+1,fin) != NULL) {
		if(cl_blankline(line)) return nseqs;
		nseqs++;
	}

	return (sint)0;	/* if you got to here-funny format/no seqs.*/
}

static sint count_msf_seqs(void)
{
/* count the number of sequences in a PILEUP alignment file */

	char line[MAXLINE+1];
	sint  nseqs;

	while (fgets(line,MAXLINE+1,fin) != NULL) {
		if(linetype(line,"//")) break;
	}

	while (fgets(line,MAXLINE+1,fin) != NULL) {
		if(!blankline(line)) break;		/* Look for next non- */
	}						/* blank line */
	nseqs = 1;

	while (fgets(line,MAXLINE+1,fin) != NULL) {
		if(blankline(line)) return nseqs;
		nseqs++;
	}

	return (sint)0;	/* if you got to here-funny format/no seqs.*/
}

static sint count_rsf_seqs(void)
{
/* count the number of sequences in a GCG RSF alignment file */

	char line[MAXLINE+1];
	sint  nseqs;

	nseqs = 0;
/* skip the comments */
	while (fgets(line,MAXLINE+1,fin) != NULL) {
 		if(line[strlen(line)-2]=='.' &&
                                 line[strlen(line)-3]=='.')
			break;
	}

	while (fgets(line,MAXLINE+1,fin) != NULL) {
                if( *line == '{' )
                      nseqs++;
	}
	return (sint)nseqs;
}

static void p_encode(char *seq, char *naseq, sint l)
{				/* code seq as ints .. use gap_pos2 for gap */
	register sint i;
/*	static char *aacids="CSTPAGNDEQHRKMILVFYW";*/
	
	for(i=1;i<=l;i++)
		if(seq[i] == '-')
			naseq[i] = gap_pos2;
		else
			naseq[i] = res_index(amino_acid_codes,seq[i]);
	naseq[i] = -3;
}

static void n_encode(char *seq,char *naseq,sint l)
{				/* code seq as ints .. use gap_pos2 for gap */
	register sint i;
/*	static char *nucs="ACGTU";	*/
	
	for(i=1;i<=l;i++) {
    	if(seq[i] == '-')          	   /* if a gap character -> code = gap_pos2 */
			naseq[i] = gap_pos2;   /* this is the code for a gap in */
		else {                     /* the input files */
			naseq[i]=res_index(amino_acid_codes,seq[i]);
		}
	}
	naseq[i] = -3;
}

static sint res_index(char *t,char c)
{
	register sint i;
	
	for(i=0;t[i] && t[i] != c;i++)
		;
	if(t[i]) return(i);
	else return -1;
}
