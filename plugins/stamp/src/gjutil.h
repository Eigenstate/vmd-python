#ifndef GJ_UTIL_H
#define GJ_UTIL_H

struct file {                    /* structure to hold a filename and */
  char *name;                    /* associated handle */
  FILE *handle;
};

struct tokens {                  /* structure to hold tokens parsed from */
  int ntok;                      /* string with strtok */
  char **tok;
};

struct std_streams {
  FILE *STDIN;
  FILE *STDOUT;
  FILE *STDERR;
};

/* utility.h function definitions */

void *GJmalloc(size_t);
void *GJrealloc(void *,size_t);
void *GJmallocNQ(size_t);
void *GJreallocNQ(void *,size_t);
void GJfree(void *);
void GJerror(const char *);
char *GJstrdup(const char *);
char *GJstoupper(const char *);
char *GJstolower(const char *);
char *GJstoup(char *);
char *GJstolo(char *);

FILE *GJfopen(const char *, const char *,int);
int  GJfclose(FILE *,int);
struct file *GJfilemake(const char *name,const char *type,int action);
struct file *GJfilerename(struct file *ret_val, const char *name);

int GJfileclose(struct file *fval,int action);
int GJfileclean(struct file *fval,int action);
void GJinitfile();

char *GJfnonnull(char *);
char *GJstrappend(char *,char *);
char *GJremovechar(char *,char);
char *GJstrcreate(size_t, char *);
char *GJstrlocate(char *,char *);
char *GJsubchar(char *,char,char);
char *GJstrtok(char *,const char *);
void error(const char *, int);
unsigned char **uchararr(int,int);
signed   char **chararr(int,int);
void GJCinit(signed char **,int ,int ,char );
void mcheck(void *, char *);
char *GJstrblank(char *, int);
void GJUCinit(unsigned char **,int ,int ,unsigned char );
char *GJcat(int N,...);
struct tokens *GJgettokens(const char *delims, char *buff);
void *GJfreetokens(struct tokens *tok);

#endif  /* GJ_UTIL_H */

