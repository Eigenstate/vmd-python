/*#include "/us1/user/julie/dmalloc/malloc.h"*/
/*********************CLUSTALW.H*********************************************/
/****************************************************************************/

   /*
   Main header file for ClustalW.  Uncomment ONE of the following 4 lines
   depending on which compiler you wish to use.
   */

/*#define VMS 1                 VAX or ALPHA VMS */

/*#define MAC 1                 Think_C for Macintosh */

/*#define MSDOS 1               Turbo C for PC's */

#define UNIX 1                /*Ultrix/Decstation, Gnu C for 
                                Sun, IRIX/SGI, OSF1/ALPHA */

/***************************************************************************/
/***************************************************************************/


#include "general.h"

#define MAXNAMES		30	/* Max chars read for seq. names */
#define MAXTITLES		60      /* Title length */
#define FILENAMELEN 	256             /* Max. file name length */
	
#define UNKNOWN   0
#define EMBLSWISS 1
#define PIR 	  2
#define PEARSON   3
#define GDE    	  4
#define CLUSTAL   5	/* DES */
#define MSF       6 /* DES */
#define RSF       7	/* JULIE */
#define USER      8	/* DES */
#define PHYLIP    9	/* DES */
#define NEXUS    10/* DES */
#define FASTA    11/* Ramu */

#define NONE      0
#define SECST     1
#define GMASK     2

#define PROFILE 0
#define SEQUENCE 1

#define BS_NODE_LABELS 2
#define BS_BRANCH_LABELS 1

#define PAGE_LEN       22   /* Number of lines of help sent to screen */

#define PAGEWIDTH	80  /* maximum characters on output file page */
#define LINELENGTH     	60  /* Output file line length */
#define GCG_LINELENGTH 	50

#ifdef VMS						/* Defaults for VAX VMS */
#define COMMANDSEP '/'
#define DIRDELIM ']'		/* Last character before file name in full file 
							   specs */
#define INT_SCALE_FACTOR 1000 /* Scaling factor to convert float to integer for profile scores */

#elif MAC
#define COMMANDSEP '/'
#define DIRDELIM ':'
#define INT_SCALE_FACTOR 100  /* Scaling factor to convert float to integer for profile scores */

#elif MSDOS
#define COMMANDSEP '/'
#define DIRDELIM '\\'
#define INT_SCALE_FACTOR 100  /* Scaling factor to convert float to integer for profile scores */

#elif UNIX
#define COMMANDSEP '-'
#define DIRDELIM '/'
#define INT_SCALE_FACTOR 1000 /* Scaling factor to convert float to integer for profile scores */
#endif

#define NUMRES 32		/* max size of comparison matrix */

#define INPUT 0
#define ALIGNED 1

#define LEFT 1
#define RIGHT 2

#define NODE 0
#define LEAF 1

#define GAPCOL 32		/* position of gap open penalty in profile */
#define LENCOL 33		/* position of gap extension penalty in profile */

typedef struct node {		/* phylogenetic tree structure */
        struct node *left;
        struct node *right;
        struct node *parent;
        float dist;
        sint  leaf;
        int order;
        char name[64];
} stree, *treeptr;

typedef struct {
	char title[30];
	char string[30];
} MatMenuEntry;

typedef struct {
	int noptions;
	MatMenuEntry opt[10];
} MatMenu;

#define MAXMAT 10

typedef struct {
	int llimit;	
	int ulimit;
	short *matptr;
	short *aa_xref;
} SeriesMat;

typedef struct {
	int nmat;
	SeriesMat mat[MAXMAT];
} UserMatSeries;
	

/*
   Prototypes
*/

/* alnscore.c */
void aln_score(void);
/* interface.c */
void parse_params(Boolean);
void init_amenu(void);
void init_interface(void);
void 	main_menu(void);
FILE 	*open_output_file(char *, char *, char *, char *);
FILE    *open_explicit_file(char *);
sint seq_input(Boolean);
Boolean open_alignment_output(char *);
void create_alignment_output(sint fseq,sint lseq);
void align(char *phylip_name);
void profile_align(char *p1_tree_name,char *p2_tree_name);/* Align 2 alignments */
void make_tree(char *phylip_name);
void get_tree(char *phylip_name);
sint profile_input(void);                        /* read a profile */
void new_sequence_align(char *phylip_name);
Boolean user_mat(char *, short *, short *);
Boolean user_mat_series(char *, short *, short *);
void get_help(char);
void clustal_out(FILE *, sint, sint, sint, sint);
void nbrf_out(FILE *, sint, sint, sint, sint);
void gcg_out(FILE *, sint, sint, sint, sint);
void phylip_out(FILE *, sint, sint, sint, sint);
void gde_out(FILE *, sint, sint, sint, sint);
void nexus_out(FILE *, sint, sint, sint, sint);
void fasta_out(FILE *, sint, sint, sint, sint);
void print_sec_struct_mask(int prf_length,char *mask,char *struct_mask);
void fix_gaps(void);


/* calcgapcoeff.c */
void calc_gap_coeff(char **alignment, sint *gaps, sint **profile, Boolean struct_penalties,
                   char *gap_penalty_mask, sint first_seq, sint last_seq,
                   sint prf_length, sint gapcoef, sint lencoef);
/* calcprf1.c */
void calc_prf1(sint **profile, char **alignment, sint *gaps, sint matrix[NUMRES ][NUMRES ], 
               sint *seq_weight, sint prf_length, sint first_seq, sint last_seq);
/* calcprf2.c */
void calc_prf2(sint **profile, char **alignment, sint *seq_weight, sint prf_length,
               sint first_seq, sint last_seq);
/* calctree.c */
void calc_seq_weights(sint first_seq, sint last_seq,sint *seq_weight);
void create_sets(sint first_seq, sint last_seq);
sint read_tree(char *treefile, sint first_seq, sint last_seq);
void clear_tree(treeptr p);
sint calc_similarities(sint nseqs);
/* clustalw.c */
int main(int argc, char **argv);
/* gcgcheck.c */
int SeqGCGCheckSum(char *seq, sint len);
/* malign.c */
sint malign(sint istart,char *phylip_name);
sint seqalign(sint istart,char *phylip_name);
sint palign1(void);
float countid(sint s1, sint s2);
sint palign2(char *p1_tree_name,char *p2_tree_name);
/* pairalign.c */
sint pairalign(sint istart, sint iend, sint jstart, sint jend);
/* prfalign.c */
lint prfalign(sint *group, sint *aligned);
/* random.c */
unsigned long linrand(unsigned long r);
unsigned long addrand(unsigned long r);
void addrandinit(unsigned long s);
/* readmat.c */
void init_matrix(void);
sint get_matrix(short *matptr, short *xref, sint matrix[NUMRES ][NUMRES ], Boolean neg_flag,
                sint scale);
sint read_user_matrix(char *filename, short *usermat, short *xref);
sint read_matrix_series(char *filename, short *usermat, short *xref);
int getargs(char *inline1, char *args[], int max);
/* sequence.c */
void fill_chartab(void);
sint readseqs(sint first_seq);
/* showpair.c */
void show_pair(sint istart, sint iend, sint jstart, sint jend);
/* trees.c */
void phylogenetic_tree(char *phylip_name,char *clustal_name,char *dist_name, char *nexus_name, char *pim_name);
void bootstrap_tree(char *phylip_name,char *clustal_name, char *nexus_name);
sint dna_distance_matrix(FILE *tree);
sint prot_distance_matrix(FILE *tree);
void guide_tree(FILE *tree,int first_seq,sint nseqs);

void calc_percidentity(FILE *pfile);

/* util.c */

void alloc_aln(sint nseqs);
void realloc_aln(sint first_seq,sint nseqs);
void free_aln(sint nseqs);
void alloc_seq(sint seq_no,sint length);
void realloc_seq(sint seq_no,sint length);
void free_seq(sint seq_no);

void *ckalloc(size_t bytes);
void *ckrealloc(void *ptr, size_t bytes);
void *ckfree(void *ptr);
char prompt_for_yes_no(char *title,char *prompt);
void fatal(char *msg, ...);
void error(char *msg, ...);
void warning(char *msg, ...);
void info(char *msg, ...);
char *rtrim(char *str);
char *blank_to_(char *str);
char *upstr(char *str);
char *lowstr(char *str);
void getstr(char *instr, char *outstr, const int SIZE);
double getreal(char *instr, double minx, double maxx, double def);
int getint(char *instr, int minx, int maxx, int def);
void do_system(void);
Boolean linetype(char *line, char *code);
Boolean keyword(char *line, char *code);
Boolean blankline(char *line);
void get_path(char *str, char *path);


