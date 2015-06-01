#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdarg.h>
#include <ctype.h>
#include "clustalw.h"

extern char **seq_array;
extern sint  *seqlen_array;
extern char **names,**titles;
extern sint *output_index;
extern sint *seq_weight;
extern double **tmat;


/*
*	ckalloc()
*
*	Tries to allocate "bytes" bytes of memory. Exits program if failed.
*	Return value:
*		Generic pointer to the newly allocated memory.
*/

void *ckalloc(size_t bytes)
{
	register void *ret;
	
	if( (ret = calloc(bytes, sizeof(char))) == NULL)
/*
	if( (ret = malloc(bytes)) == NULL)
*/
		fatal("Out of memory\n");
	else
		return ret;	

	return ret;	
}

/*
*	ckrealloc()
*
*	Tries to reallocate "bytes" bytes of memory. Exits program if failed.
*	Return value:
*		Generic pointer to the re-allocated memory.
*/

void *ckrealloc(void *ptr, size_t bytes)
{
	register void *ret=NULL;

	if (ptr == NULL)	
		fatal("Bad call to ckrealloc\n");
	else if( (ret = realloc(ptr, bytes)) == NULL)
		fatal("Out of memory\n");
	else
		return ret;	

	return ret;	
}

/*
*	ckfree()
*
*	Tries to free memory allocated by ckalloc.
*	Return value:
*		None.
*/

void *ckfree(void *ptr)
{
	if (ptr == NULL)
		warning("Bad call to ckfree\n");
	else {
	 	free(ptr);
		ptr = NULL;
	}
	return ptr;
}


/*
*	rtrim()
*
*	Removes trailing blanks from a string
*
*	Return values:
*		Pointer to the processed string
*/

char * rtrim(char *str)
{
	register int p;

	p = strlen(str) - 1;
	
	while ( isspace(str[p]) )
		p--;
		
	str[p + 1] = EOS;
	
	return str;
}


/*
*	blank_to_()
*
*	Replace blanks in a string with underscores
*
*       Also replaces , ; : ( or ) with _
*
*	Return value:
*		Pointer to the processed string
*/

char * blank_to_(char *str)
{
	int i,p;

	p = strlen(str) - 1;
	
	for(i=0;i<=p;i++) 
		if(
                     (str[i]==' ') ||
                     (str[i]==';') ||
                     (str[i]==',') ||
                     (str[i]=='(') ||
                     (str[i]==')') ||
                     (str[i]==':') 
                  )
                      str[i] = '_';
	
	return str;
}


/*
*	upstr()
*
*	Converts string str to uppercase.
*	Return values:
*		Pointer to the converted string.
*/

char * upstr(char *str)
{
	register char *s = str;
	
	while( (*s = toupper(*s)) )
		s++;
		
	return str;
}

/*
*	lowstr()
*
*	Converts string str to lower case.
*	Return values:
*		Pointer to the converted string.
*/

char * lowstr(char *str)
{
	register char *s = str;
	
	while( (*s = tolower(*s)) )
		s++;
		
	return str;
}

void getstr(char *instr,char *outstr, const int SIZE)
{	
	fprintf(stdout,"%s: ",instr);
	fgets(outstr, SIZE, stdin);
}

double getreal(char *instr,double minx,double maxx,double def)
{
	int status;
	float ret;
	char line[MAXLINE+1];	
	
	while(TRUE) {
		fprintf(stdout,"%s (%.1f-%.1f)   [%.1f]: ",instr,minx,maxx,def);
		fgets(line, MAXLINE, stdin);
		status=sscanf(line,"%f",&ret);
		if(status == EOF) return def;
		if(ret>maxx) {
			fprintf(stdout,"ERROR: Max. value=%.1f\n\n",maxx);
			continue;
		}
		if(ret<minx) {
			fprintf(stdout,"ERROR: Min. value=%.1f\n\n",minx);
			continue;
		}
		break;
	}
	return (double)ret;
}


int getint(char *instr,int minx,int maxx, int def)
{
	int ret,status;
	char line[MAXLINE+1];	

	while(TRUE) {
		fprintf(stdout,"%s (%d..%d)    [%d]: ",
		instr,(pint)minx,(pint)maxx,(pint)def);
		fgets(line, MAXLINE, stdin);
		status=sscanf(line,"%d",&ret);
		if(status == EOF) return def;
		if(ret>maxx) {
			fprintf(stdout,"ERROR: Max. value=%d\n\n",(pint)maxx);
			continue;
		}
		if(ret<minx) {
			fprintf(stdout,"ERROR: Min. value=%d\n\n",(pint)minx);
			continue;
		}
		break;
	}
	return ret;
}

void do_system(void)
{
	char line[MAXLINE+1];
	
	getstr("\n\nEnter system command",line, MAXLINE);
	if(*line != EOS)
		system(line);
	fprintf(stdout,"\n\n");
}


Boolean linetype(char *line,char *code)
{
	return( strncmp(line,code,strlen(code)) == 0 );
}

Boolean keyword(char *line,char *code)
{
	int i;
	char key[MAXLINE];

	for(i=0;!isspace(line[i]) && line[i]!=EOS;i++)
		key[i]=line[i];
	key[i]=EOS;
	return( strcmp(key,code) == 0 );
}

Boolean blankline(char *line)
{
	int i;

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


void get_path(char *str,char *path)
{
	register int i;
	
	strcpy(path,str);
	for(i=strlen(path)-1;i>-1;--i) {
		if(str[i]==DIRDELIM) {
			i = -1;
			break;
		}
		if(str[i]=='.') break;
	}
	if(i<0)
		strcat(path,".");
	else
		path[i+1]=EOS;
}

void alloc_aln(sint nseqs)
{
	sint i,j;

        seqlen_array = (sint *)ckalloc( (nseqs+1) * sizeof (sint));

        seq_array = (char **)ckalloc( (nseqs + 1) * sizeof (char *) );
	for(i=0;i<nseqs+1;i++)
		seq_array[i]=NULL;

        names = (char **)ckalloc( (nseqs+1) * sizeof (char *) );
        for(i=1;i<=nseqs;i++)
                names[i] = (char *)ckalloc((MAXNAMES+1) * sizeof (char));

        titles = (char **)ckalloc( (nseqs+1) * sizeof (char *) );
        for(i=1;i<=nseqs;i++)
                titles[i] = (char *)ckalloc((MAXTITLES+1) * sizeof (char));

        output_index = (sint *)ckalloc( (nseqs+1) * sizeof (sint));

        tmat = (double **) ckalloc( (nseqs+1) * sizeof (double *) );
        for(i=1;i<=nseqs;i++)
                tmat[i] = (double *)ckalloc( (nseqs+1) * sizeof (double) );
        for(i=1;i<=nseqs;i++)
        	for(j=1;j<=nseqs;j++)
			tmat[i][j]=0.0;

        seq_weight = (sint *)ckalloc( (nseqs+1) * sizeof (sint));
        for(i=1;i<=nseqs;i++)
		seq_weight[i]=100;
}

void realloc_aln(sint first_seq,sint nseqs)
{
	sint i,j;

        seqlen_array = (sint *)ckrealloc(seqlen_array, (first_seq+nseqs+1) * sizeof (sint));

        seq_array = (char **)ckrealloc(seq_array, (first_seq+nseqs+1) * sizeof (char *) );
	for(i=first_seq;i<first_seq+nseqs+1;i++)
		seq_array[i]=NULL;

        names = (char **)ckrealloc(names, (first_seq+nseqs+1) * sizeof (char *) );
        for(i=first_seq;i<first_seq+nseqs;i++)
                names[i] = (char *)ckalloc((MAXNAMES+1) * sizeof (char));

        titles = (char **)ckrealloc(titles, (first_seq+nseqs+1) * sizeof (char *) );
        for(i=first_seq;i<first_seq+nseqs;i++)
                titles[i] = (char *)ckalloc((MAXTITLES+1) * sizeof (char));

        output_index = (sint *)ckrealloc(output_index, (first_seq+nseqs+1) * sizeof (sint));

        seq_weight = (sint *)ckrealloc(seq_weight, (first_seq+nseqs+1) * sizeof (sint));
        for(i=first_seq;i<first_seq+nseqs;i++)
		seq_weight[i]=100;

        tmat = (double **) ckrealloc(tmat, (first_seq+nseqs+1) * sizeof (double *) );
        for(i=1;i<first_seq;i++)
                tmat[i] = (double *)ckrealloc(tmat[i], (first_seq+nseqs+1) * sizeof (double) );
        for(i=first_seq;i<first_seq+nseqs;i++)
                tmat[i] = (double *)ckalloc( (first_seq+nseqs+1) * sizeof (double) );
        for(i=1;i<first_seq;i++)
        	for(j=first_seq;j<first_seq+nseqs;j++)
		{
			tmat[i][j]=0.0;
			tmat[j][i]=0.0;
		}
}

void free_aln(sint nseqs)
{
	sint i;

	if(nseqs<=0) return;

	seqlen_array = ckfree(seqlen_array);

        for(i=1;i<=nseqs;i++)
		seq_array[i] = ckfree(seq_array[i]);
	seq_array = ckfree(seq_array);

	for(i=1;i<=nseqs;i++)
		names[i] = ckfree(names[i]);
	names = ckfree(names);

	for(i=1;i<=nseqs;i++)
		titles[i] = ckfree(titles[i]);
	titles = ckfree(titles);

	output_index = ckfree(output_index);

	seq_weight = ckfree(seq_weight);

        for(i=1;i<=nseqs;i++)
		tmat[i] = ckfree(tmat[i]);
	tmat = ckfree(tmat);
}

void alloc_seq(sint seq_no,sint length)
{
	seq_array[seq_no] = (char *)ckalloc((length+2) * sizeof (char));
}

void realloc_seq(sint seq_no,sint length)
{
	seq_array[seq_no] = (char *)realloc(seq_array[seq_no], (length+2) * sizeof (char));

}

void free_seq(sint seq_no)
{
	seq_array[seq_no]=ckfree(seq_array[seq_no]);
}

