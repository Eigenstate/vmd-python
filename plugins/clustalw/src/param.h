#define MAXARGS 100

typedef struct {
	char *str;
	sint *flag;
	int type;
	char **arg;
} cmd_line_data;

/* 
   command line switches
*/
sint setoptions = -1;
sint sethelp = -1;
sint setinteractive = -1;
sint setbatch = -1;
sint setgapopen = -1;
sint setgapext = -1;
sint setpwgapopen = -1;
sint setpwgapext = -1;
sint setoutorder = -1;
sint setbootlabels = -1;
sint setpwmatrix = -1;
sint setmatrix = -1;
sint setpwdnamatrix = -1;
sint setdnamatrix = -1;
sint setnegative = -1;
sint setnoweights = -1;
sint setoutput = -1;
sint setoutputtree = -1;
sint setquicktree = -1;
sint settype = -1;
sint setcase = -1;
sint setseqno = -1;

sint setseqno_range = -1;
sint setrange = -1;

sint settransweight = -1;
sint setseed = -1;
sint setscore = -1;
sint setwindow = -1;
sint setktuple = -1;
sint setkimura = -1;
sint settopdiags = -1;
sint setpairgap = -1;
sint settossgaps = -1;
sint setnopgap = -1;
sint setnohgap = -1;
sint setnovgap = -1;
sint sethgapres = -1;
sint setvgapres = -1;
sint setuseendgaps = -1;
sint setmaxdiv = -1;
sint setgapdist = -1;
sint setdebug = -1;
sint setoutfile = -1;
sint setinfile = -1;
sint setprofile1 = -1;
sint setprofile2 = -1;
sint setalign = -1;
sint setconvert = -1;
sint setnewtree = -1;
sint setusetree = -1;
sint setnewtree1 = -1;
sint setusetree1 = -1;
sint setnewtree2 = -1;
sint setusetree2 = -1;
sint setbootstrap = -1;
sint settree = -1;
sint setprofile = -1;
sint setsequences = -1;
sint setsecstr1 = -1;
sint setsecstr2 = -1;
sint setsecstroutput = -1;
sint sethelixgap = -1;
sint setstrandgap = -1;
sint setloopgap = -1;
sint setterminalgap = -1;
sint sethelixendin = -1;
sint sethelixendout = -1;
sint setstrandendin = -1;
sint setstrandendout = -1;

/*
   multiple alignment parameters
*/
float 		dna_gap_open = 15.0,  dna_gap_extend = 6.66;
float 		prot_gap_open = 10.0, prot_gap_extend = 0.2;
sint		profile_type = PROFILE;
sint 		gap_dist = 4;
sint 		output_order   = ALIGNED;
sint    	divergence_cutoff = 30;
sint	    matnum = 3;
char 		mtrxname[FILENAMELEN+1] = "gonnet";
sint	    dnamatnum = 1;
char 		dnamtrxname[FILENAMELEN+1] = "iub";
char 		hyd_residues[] = "GPSNDQEKR";
Boolean 	no_weights = FALSE;
Boolean 	neg_matrix = FALSE;
Boolean		no_hyd_penalties = FALSE;
Boolean		no_var_penalties = TRUE;
Boolean		no_pref_penalties = FALSE;
Boolean		use_endgaps = FALSE;
Boolean		endgappenalties = FALSE;
Boolean		reset_alignments_new  = FALSE;		/* DES */
Boolean		reset_alignments_all  = FALSE;		/* DES */
sint		output_struct_penalties = 0;
sint        struct_penalties1 = NONE;
sint        struct_penalties2 = NONE;
Boolean		use_ss1 = TRUE;
Boolean		use_ss2 = TRUE;
sint        helix_penalty = 4;
sint        strand_penalty = 4;
sint        loop_penalty = 1;
sint        helix_end_minus = 3;
sint        helix_end_plus = 0;
sint        strand_end_minus = 1;
sint        strand_end_plus = 1;
sint        helix_end_penalty = 2;
sint        strand_end_penalty = 2;
Boolean	    use_ambiguities = FALSE;

/*
   pairwise alignment parameters
*/
float  		dna_pw_go_penalty = 15.0,  dna_pw_ge_penalty = 6.66;
float 		prot_pw_go_penalty = 10.0, prot_pw_ge_penalty = 0.1;
sint	    pw_matnum = 3;
char 		pw_mtrxname[FILENAMELEN+1] = "gonnet";
sint	    pw_dnamatnum = 1;
char 		pw_dnamtrxname[FILENAMELEN+1] = "iub";
char     usermtrxname[FILENAMELEN+1], pw_usermtrxname[FILENAMELEN+1];
char     dnausermtrxname[FILENAMELEN+1], pw_dnausermtrxname[FILENAMELEN+1];

Boolean  	quick_pairalign = FALSE;
float		transition_weight = 0.5;
sint		new_seq;

/*
   quick pairwise alignment parameters
*/
sint   	     	dna_ktup      = 2;   /* default parameters for DNA */
sint    	    	dna_wind_gap  = 5;
sint    	    	dna_signif    = 4;
sint    	    	dna_window    = 4;

sint        	prot_ktup     = 1;   /* default parameters for proteins */
sint        	prot_wind_gap = 3;
sint        	prot_signif   = 5;
sint        	prot_window   = 5;
Boolean         percent=TRUE;
Boolean		tossgaps = FALSE;
Boolean		kimura = FALSE;


sint	        boot_ntrials  = 1000;
unsigned sint    boot_ran_seed = 111;


sint    		debug = 0;

Boolean        	explicit_dnaflag = FALSE; /* Explicit setting of sequence type on comm.line*/
Boolean        	lowercase = TRUE; /* Flag for GDE output - set on comm. line*/
Boolean        	cl_seq_numbers = FALSE;

Boolean        	seqRange = FALSE; /* Ramu */

Boolean        	output_clustal = TRUE;
Boolean        	output_gcg     = FALSE;
Boolean        	output_phylip  = FALSE;
Boolean        	output_nbrf    = FALSE;
Boolean        	output_gde     = FALSE;
Boolean        	output_nexus   = FALSE;
Boolean        	output_fasta   = FALSE;

Boolean         showaln        = TRUE;
Boolean         save_parameters = FALSE;

/* DES */
Boolean        	output_tree_clustal   = FALSE;
Boolean        	output_tree_phylip    = TRUE;
Boolean        	output_tree_distances = FALSE;
Boolean        	output_tree_nexus = FALSE;
Boolean        	output_pim = FALSE;


sint		bootstrap_format      = BS_BRANCH_LABELS;

/*These are all the positively scoring groups that occur in the Gonnet Pam250
matrix. There are strong and weak groups, defined as strong score >0.5 and
weak score =<0.5. Strong matching columns to be assigned ':' and weak matches
assigned '.' in the clustal output format.
*/

char *res_cat1[] = {
                "STA",
                "NEQK",
                "NHQK",
                "NDEQ",
                "QHRK",
                "MILV",
                "MILF",
                "HY",
                "FYW",
                NULL };

char *res_cat2[] = {
                "CSA",
                "ATV",
                "SAG",
                "STNK",
                "STPA",
                "SGND",
                "SNDEQK",
                "NDEQHK",
                "NEQHRK",
                "FVLIM",
                "HFY",
                NULL };



static char *type_arg[] = {
                "protein",
                "dna",
		""};

static char *bootlabels_arg[] = {
                "node",
                "branch",
		""};

static char *outorder_arg[] = {
                "input",
                "aligned",
		""};

static char *case_arg[] = {
                "lower",
                "upper",
		""};

static char *seqno_arg[] = {
                "off",
                "on",
		""};

static char *seqno_range_arg[] = {
                "off",
                "on",
		""};

static char *score_arg[] = {
                "percent",
                "absolute",
		""};

static char *output_arg[] = {
                "gcg",
                "gde",
                "pir",
                "phylip",
                "nexus",
                "fasta",
		""};

static char *outputtree_arg[] = {
                "nj",
                "phylip",
                "dist",
                "nexus",
		""};

static char *outputsecstr_arg[] = {
                "structure",
                "mask",
                "both",
                "none",
		""};

/*
     command line initialisation

     type = 0    no argument
     type = 1    integer argument
     type = 2    float argument
     type = 3    string argument
     type = 4    filename
     type = 5    opts
*/
#define NOARG 0
#define INTARG 1
#define FLTARG 2
#define STRARG 3
#define FILARG 4
#define OPTARG 5


/* command line switches for DATA       **************************/
cmd_line_data cmd_line_file[] = {
     {"infile",		&setinfile,		FILARG,	NULL},
     {"profile1",	&setprofile1,		FILARG,	NULL},
     {"profile2",	&setprofile2,		FILARG,	NULL},
     {"",		NULL,			-1, NULL}};
/* command line switches for VERBS      **************************/
cmd_line_data cmd_line_verb[] = {
     {"help",		&sethelp,		NOARG,	NULL},
     {"check",       &sethelp,    		NOARG,	NULL},
     {"options",		&setoptions,		NOARG,	NULL},
     {"align",		&setalign,		NOARG,	NULL},
     {"newtree",		&setnewtree,		FILARG,	NULL},
     {"usetree",		&setusetree,		FILARG,	NULL},
     {"newtree1",	&setnewtree1,		FILARG,	NULL},
     {"usetree1",	&setusetree1,		FILARG,	NULL},
     {"newtree2",	&setnewtree2,		FILARG,	NULL},
     {"usetree2",	&setusetree2,		FILARG,	NULL},
     {"bootstrap",	&setbootstrap,		NOARG,	NULL},
     {"tree",		&settree, 		NOARG,	NULL},
     {"quicktree",	&setquicktree,		NOARG,	NULL},
     {"convert",		&setconvert,		NOARG,	NULL},
     {"interactive",	&setinteractive,	NOARG,	NULL},
     {"batch",		&setbatch,		NOARG,	NULL},
     {"",		NULL,			-1, NULL}};
/* command line switches for PARAMETERS **************************/
cmd_line_data cmd_line_para[] = {
     {"type",		&settype,		OPTARG,	type_arg},
     {"profile",	&setprofile,	NOARG,	NULL},
     {"sequences",	&setsequences,	NOARG,	NULL},
     {"matrix",		&setmatrix,		FILARG,	NULL},
     {"dnamatrix",	&setdnamatrix,		FILARG,	NULL},
     {"negative",	&setnegative,		NOARG,	NULL},
     {"noweights",	&setnoweights,		NOARG,	NULL},
     {"gapopen", 	&setgapopen,		FLTARG,	NULL},
     {"gapext",		&setgapext,		FLTARG,	NULL},
     {"endgaps",		&setuseendgaps,		NOARG,	NULL},
     {"nopgap",		&setnopgap,		NOARG,	NULL},
     {"nohgap",		&setnohgap,		NOARG,	NULL},
     {"novgap",		&setnovgap,		NOARG,	NULL},
     {"hgapresidues",	&sethgapres,		STRARG,	NULL},
     {"maxdiv",		&setmaxdiv,		INTARG,	NULL},

     {"gapdist",		&setgapdist,		INTARG,	NULL},
     {"pwmatrix",	&setpwmatrix,		FILARG,	NULL},
     {"pwdnamatrix",	&setpwdnamatrix,	FILARG,	NULL},
     {"pwgapopen",	&setpwgapopen,		FLTARG,	NULL},
     {"pwgapext",	&setpwgapext,		FLTARG,	NULL},
     {"ktuple",		&setktuple,		INTARG,	NULL},
     {"window",		&setwindow,		INTARG,	NULL},
     {"pairgap",		&setpairgap,		INTARG,	NULL},
     {"topdiags",	&settopdiags,		INTARG,	NULL},
     {"score",		&setscore,		OPTARG,	score_arg},
     {"transweight",	&settransweight,	FLTARG,	NULL},
     {"seed",		&setseed,		INTARG,	NULL},
     {"kimura",		&setkimura,		NOARG,	NULL},
     {"tossgaps",	&settossgaps,		NOARG,	NULL},
     {"bootlabels",	&setbootlabels,		OPTARG,	bootlabels_arg},
     {"debug",		&setdebug,		INTARG,	NULL},
     {"output",		&setoutput,		OPTARG,	output_arg},
     {"outputtree",	&setoutputtree,		OPTARG,	outputtree_arg},
     {"outfile",		&setoutfile,		FILARG,	NULL},
     {"outorder",	&setoutorder,		OPTARG,	outorder_arg},
     {"case",		&setcase,		OPTARG,	case_arg},
     {"seqnos",		&setseqno,		OPTARG,	seqno_arg},

     {"seqno_range",	&setseqno_range,	OPTARG,	seqno_range_arg}, /* this one should be on/off  and */
     {"range",           &setrange,             STRARG, NULL},  /* this one should be like 10:20  ,   messy option settings */

     {"nosecstr1",   &setsecstr1,		NOARG, NULL},
     {"nosecstr2",   &setsecstr2,		NOARG, NULL},
     {"secstrout",   &setsecstroutput,	OPTARG,  outputsecstr_arg},
     {"helixgap",    &sethelixgap,		INTARG, NULL},
     {"strandgap",   &setstrandgap,		INTARG, NULL},
     {"loopgap",     &setloopgap,		INTARG, NULL},
     {"terminalgap", &setterminalgap,	INTARG, NULL},
     {"helixendin",  &sethelixendin,		INTARG, NULL},
     {"helixendout", &sethelixendout,	INTARG, NULL},
     {"strandendin", &setstrandendin,	INTARG, NULL},
     {"strandendout",&setstrandendout,	INTARG, NULL},

     {"",		NULL,			-1, NULL}};


