void rna(char *pdbfile, long *type_stat,long **pair_stat, long *base_all);
void work_horse(char *pdbfile, FILE *fout, long num_residue, long num,
                char *bseq, long **seidx, long *RY, char **AtomName,
                char **ResName, char *ChainID, long *ResSeq,char **Miscs, 
                double **xyz,long num_modify, long *modify_idx, 
                long *type_stat,long **pair_stat);
void RY_edge_stat(char *pdbfile, long np);

void print_sorted_pair(long ntot, char *pdbfile);
void sort_by_pair(FILE *, long n, char **str);
void ring_center(long i,long **seidx,char *bseq,char **AtomName, 
			double **xyz, double *xyz_c);
void cc_distance(long i, long j, char *bseq, long **seidx,
                 char **AtomName, double **xyz, double *);

void base_stack(long i, long j, char *bseq, long **seidx, char **AtomName,
                double **xyz, double *rtn_val, long *yes);

void check_base_base(long i, long j, double hlimit, long **seidx,
                     char **AtomName, double **xyz, long *yes);
void get_unequility(long num_hbonds, char **hb_atom1, long *nh, char **atom);

void pair_type_statistics(FILE *fout, long num_pair_tot, char **pair_type,
                          long *type_stat);
void rna_select(char *pdbfile, double resolution, long *yes);

void write_base_pair_xml(FILE *xml, char *parfile, char chain_nam1,char chain_nam2,
                         char *tab5, char *tab6, char *tab7);

void write_base_pair_mol(FILE *xml, long molID, char *parfile, long *chain_res, 
                         char *tag5, char *tag6, char *tag7);

void write_base_pair_int(FILE *xml, long i,long j, char *parfile, long *chain_res);


void write_helix_mol(FILE *xml, long chain1, long chain2,long *chain_res,
                     long xml_nh, long **xml_helix,long *xml_helix_len);


void get_residue_work_num(char *str, long *nres1, long *nres2);

void check_nmr_xray(char *inpfile, long *key, char *outfile);
void check_model(FILE *fp, long *yes);
void extract_nmr_coord(char *inpfile,  char *outfile);
void get_chain_idx(long num_residue, long **seidx, char *ChainID, 
                   long *nchain, long **chain_idx);
void xml_molecule(long molID,long *chain_res, long **chain_idx, char chain_nam, char *bseq,
                  long **seidx,char **AtomName, char **ResName, char *ChainID,
                  long *ResSeq, char **Miscs, double **xyz, long xml_nh,
                  long **xml_helix, long *xml_helix_len, long xml_ns,
                  long *xml_bases, double **base_xy, long num_modify,
                  long *modify_idx,long num_loop, long **loop,
                  long **sing_st_end,  char *parfile, long *sugar_syn,FILE *xml);
void xml_interaction(long i,long j, char chain_nam1, char chain_nam2,
                     long xml_nh, long **xml_helix, long *xml_helix_len,
                     long **seidx,char *ChainID,char *parfile, FILE *xml);

void xml2ps(char *pdbfile, long nres, long XML);

void write_multiplets(char *pdbfile);
/*void token_str(char str[], char token[], int *nstr, char line[][80]);*/
long read_pdb_ref(char *pdbfile, char **sAtomName, double **sxyz);
char identify_uncommon(long , char **AtomName, long ib, long ie);
void get_reference_pdb(char *BDIR);
long  ref_idx(char resnam);
void type_k1_K2(long k1, long k2, long num_pair_tot, long **bs_pairs_tot,
                char **pair_type, char *str);
void write_tmp_pdb(char *pdbfile,long nres, long **seidx, char **AtomName,
                   char **ResName,char *ChainID, long *ResSeq, double **xyz);
void ps_label_chain(FILE *psfile, long num_residue, char *bseq,long **seidx,
                    char *ChainID, double **xy_bs);
void ps_label_residue(FILE *psfile, long num_residue, char *bseq,long **seidx,
                      char *ChainID, char **ResName, double **xy_bs);

void  base_suger_only(long i, long j, double *HB_UPPER, long **seidx,
                      char **AtomName, char *HB_ATOM, double **xyz,
                      long *key1, long *key2 );


void get_hbond(long i, long j, double *HB_UPPER, long **seidx,
               char **AtomName, char *HB_ATOM, double **xyz,
               long *nh, char **hb_atom1, char **hb_atom2, double *hb_dist);

void check_pairs(long i, long j, char *bseq, long **seidx, double **xyz,
                 double **Nxyz, double **orien, double **org, char **AtomName,
                 double *BPRS, double *rtn_val, long *bpid, long network);

void base_base_dist(long i, long j, long **seidx, char **AtomName, char *bseq,
                     double **xyz, double dist,  long *nh, char **hb_atom1,
                     char **hb_atom2, double *hb_dist);

void check_pair_lu(long i, long j, char *bseq, long **seidx, double **xyz,
                double **Nxyz, double **orien, double **org, char **AtomName,
                double *BPRS, double *rtn_val, long *bpid, long network, long *RY);
void single_BB_Hbond(long i, long j, long **seidx, char **AtomName, char *bseq,
                     double **xyz, long *Hyes);        
void non_Hbond_pair(long i, long j, long m, long n, char **AtomName,
                    long *RY, long *yes);

void H_catalog(long i,long m, char *bseq, char **AtomName,
               long *without_H,long *with_H);
void Hbond_pair(long i, long j, long **seidx, char **AtomName, char *bseq,
                double **xyz, double change,  long *nh, char **hb_atom1,
                char **hb_atom2, double *hb_dist, long c_key, long bone_key);

void motif(char *pdbfile);

void get_max_npatt(long np, long *npatt,  long *max_npatt);
long nline(char  *inpfile);
void get_str(char *inpfile, long *nline, char **str_pair); 
void token_str(char *str, char *token, long *nstr, char **line); 
void get_working_num(char *str, char *working_num);
void syn_or_anti( long num_residue, char **AtomName, long **seidx,
                  double **xyz, long *RY, long *sugar_syn);



/* analyze.c */
void bp_analyze(char *pdbfile,long num,char **AtomName, char **ResName, char *ChainID, long *ResSeq,
                double **xyz, long num_residue,  char **Miscs, long **seidx,
                long num_bp, long **pair_num);
void print_header(long num_bp,  long num_residue, char *pdbfile, FILE * fp);

void get_bpseq(long ds, long num_bp, long **pair_num, long **seidx,
               char **AtomName, char **ResName, char *ChainID,long *ResSeq,
               char **Miscs, double **xyz, char **bp_seq, long **RY);
void pair_checking(long ip, long ds, long num_residue, char *pdbfile,
                   long *num_bp, long **pair_num);
void atom_list(long ds, long num_bp, long **pair_num, long **seidx,
               long **RY, char **bp_seq, char **AtomName, char **ResName,
               char *ChainID, long *ResSeq, char **Miscs, long **phos,
               long **c6_c8, long **backbone, long **sugar, long **chi);
void parcat(char *str, double par, char *format, char *bstr);
void backbone_torsion(long ds, long num_bp, char **bp_seq, long **backbone,
                      long **sugar, long **chi, double **xyz, FILE * fp);
void p_c1_dist(long ds, long num_bp, char **bp_seq, long **phos,
               long **chi, double **xyz, FILE * fp);
void lambda_d3(long num_bp, char **bp_seq, long **chi, long **c6_c8,
               double **xyz, FILE * fp);
void print_axyz(long num_bp, char **bp_seq, long **aidx, char *aname,
                double **xyz);
double gw_angle(long *idx, double *pvec12, double **xyz);
void groove_width(long num_bp, char **bp_seq, long **phos, double **xyz,
                  FILE * fp);
void ref_frames(long ds, long num_bp, long **pair_num, char **bp_seq,
                long **seidx, long **RY, char **AtomName, char **ResName,
                char *ChainID, long *ResSeq, char **Miscs, double **xyz,
                FILE * fp, double **orien, double **org, long *WC_info,
                long *str_type);
void hb_information(long num_bp, long **pair_num, char **bp_seq,
                    long **seidx, char **AtomName, double **xyz,
                    long *WC_info, FILE * fp);
void helical_par(double **rot1, double *org1, double **rot2, double *org2,
                 double *pars, double **mst_orien, double *mst_org);
void print_par(char **bp_seq, long num_bp, long ich, long ishel,
               double **param, FILE * fp);
void single_helix(long num_bp, char **bp_seq, double **step_par,
                  double **heli_par, double **orien, double **org, FILE * fp);
void double_helix(long num_bp, char **bp_seq, double **step_par,
                  double **heli_par, double **orien, double **org,
                  FILE * fp, double *twist, double *mst_orien,
                  double *mst_org, double *mst_orienH, double *mst_orgH);
void get_parameters(long ds, long num_bp, char **bp_seq, double **orien,
                    double **org, FILE * fp, double *twist,
                    double *mst_orien, double *mst_org, double *mst_orienH,
                    double *mst_orgH);
void vec2mtx(double *parvec, long num, double **parmtx);
void print_ss_rebuild_pars(double **pars, long num_bp, char *str,
                           char **bp_seq, FILE * fp);
void print_ds_rebuild_pars(double **bp_par, double **step_par, long num_bp,
                           char *str, char **bp_seq, FILE * fp);
void print_ref(char **bp_seq, long num_bp, long ich, double *org,
               double *orien, FILE * fp);
void write_mst(long num_bp, long **pair_num, char **bp_seq,
               double *mst_orien, double *mst_org, long **seidx,
               char **AtomName, char **ResName, char *ChainID,
               long *ResSeq, double **xyz, char **Miscs, char *strfile);
void print_xyzP(long nbpm1, char **bp_seq, long **phos, double *mst_orien,
                double *mst_org, double **xyz, FILE * fp, char *title_str,
                double **aveP);
void print_PP(double mtwist, double *twist, long num_bp, char **bp_seq, long **phos,
              double *mst_orien, double *mst_org, double *mst_orienH,
            double *mst_orgH, double **xyz, long *WC_info, long *bphlx, FILE * fp);
void str_classify(double mtwist, long str_type, long num_bp, FILE * fp);
double a_hlxdist(long idx, double **xyz, double *hlx_axis, double *hlx_pos);
void print_radius(char **bp_seq, long nbpm1, long ich, double **p_radius,
                  double **o4_radius, double **c1_radius, FILE * fp);
void helix_radius(long ds, long num_bp, char **bp_seq, double **orien,
                  double **org, long **phos, long **chi, double **xyz,
                  FILE * fp);
void print_shlx(char **bp_seq, long nbpm1, long ich, double *shlx_orien,
                double *shlx_org, FILE * fp);
void helix_axis(long ds, long num_bp, char **bp_seq, double **orien,
                double **org, FILE * fp);
void refs_right_left(long bnum, double **orien, double **org,
                     double **r1, double *o1, double **r2, double *o2);
void refs_i_ip1(long bnum, double *bp_orien, double *bp_org,
                double **r1, double *o1, double **r2, double *o2);
void print_analyze_ref_frames(long ds, long num_bp, char **bp_seq,
                              double *iorg, double *iorien);
void get_helix(long ds, long num_bp, long num, long **chi, double **xyz,
               double *std_rise, double *hrise, double *haxis,
               double *hstart, double *hend);
void global_analysis(long ds, long num_bp, long num, char **bp_seq,
                     long **chi, long **phos, double **xyz, FILE * fp);

void get_hbond_ij_LU(long i, long j, double *HB_UPPER, long **seidx,
                  char **AtomName, char *HB_ATOM, double **xyz, char *HB_INFO);
void change_xyz(long side_view, double *morg, double **mst, long num, double **xyz);
  void get_side_view(long ib, long ie, double **xyz);
void r3d_rod(long itype, double *xyz1, double *xyz2, double rad, double *rgbv, FILE * fp);
void get_r3dpars(double **base_col, double *hb_col, double *width3,
                 double **atom_col, char *label_style);

void compdna(double **rot1, double *org1, double **rot2, double *org2,
             double *pars, double **mst_orien, double *mst_org);

void curves(double **rot1, double *org1, double **rot2, double *org2,
            double *pars);

void curves_mbt(long ibp, double **orien, double **org, double **cvr,
                double *cvo);

void freehelix(double **rot1, double *org1, double **rot2, double *org2,
               double *pars, double **mst_orien, double *mst_org);

void sgl_helix(double **rot1, double **rot2, double *rot_ang,
               double *rot_hlx);

void ngeom(double **rot1, double *org1, double **rot2, double *org2,
           double *pars, double **mst_orien, double *mst_org);

void nuparm(double **rot1, double *org1, double **rot2, double *org2,
            double *pars, double **mst_orien, double *mst_org,
            double *hpars, long get_hpar);

void rna_lu(double **rot1, double *org1, double *pvt1, double **rot2,
         double *org2, double *pvt2, double *pars, double **mst_orien,
         double *mst_org);
void cehs_bppar(double **rot1, double *org1, double **rot2, double *org2,
                double *pars, double **mst_orien, double *mst_org);

void other_pars(long num_bp, char **bp_seq, double **orien, double **org);

/* cmn_fncs.c */
void dswap(double *pa, double *pb);
void lswap(long *pa, long *pb);
double dmax(double a, double b);
double dmin(double a, double b);
double ddiff(double a, double b);
FILE *open_file(char *filename, char *filemode);
long close_file(FILE * fp);
long upperstr(char *a);
long number_of_atoms(char *pdbfile);
long read_pdb(char *pdbfile, char **AtomName, char **ResName, char *ChainID,
              long *ResSeq, double **xyz, char **Miscs, char *ALT_LIST);
void pdb_record(long ib, long ie, long *inum, long idx, char **AtomName,
                char **ResName, char *ChainID, long *ResSeq, double **xyz,
                char **Miscs, FILE * fp);
void write_pdb(long num, char **AtomName, char **ResName, char *ChainID,
               long *ResSeq, double **xyz, char **Miscs, char *pdbfile);
void write_pdbcnt(long num, char **AtomName, char **ResName, char *ChainID,
     long *ResSeq, double **xyz, long nlinked_atom, long **connect, char *pdbfile);
void max_dmatrix(double **d, long nr, long nc, double *maxdm);
void min_dmatrix(double **d, long nr, long nc, double *mindm);
void ave_dmatrix(double **d, long nr, long nc, double *avedm);
void std_dmatrix(double **d, long nr, long nc, double *stddm);
double max_dvector(double *d, long n);
double min_dvector(double *d, long n);
double ave_dvector(double *d, long n);
double std_dvector(double *d, long n);
void move_position(double **d, long nr, long nc, double *mpos);
void print_sep(FILE * fp, char x, long n);
long **residue_idx(long num, long *ResSeq, char **Miscs, char *ChainID,
                   char **ResName, long *num_residue);
long residue_ident(char **AtomName, double **xyz, long ib, long ie);
char base_ident(FILE *fout,char *rname, char *idmsg, char **baselist,
                long num_sb);
void get_baselist(char **baselist, long *num_sb);
void get_seq(FILE *fout, long num_residue, long **seidx, char **AtomName,
             char **ResName, char *ChainID, long *ResSeq, char **Miscs,
             double **xyz, char *bseq, long *RY, long *num_modify, long *modify_idx);
void get_bpseq(long ds, long num_bp, long **pair_num, long **seidx,
               char **AtomName, char **ResName, char *ChainID,
               long *ResSeq, char **Miscs, double **xyz, char **bp_seq, long **RY);
long num_strmatch(char *str, char **strmat, long nb, long ne);
long find_1st_atom(char *str, char **strmat, long nb, long ne, char *idmsg);
void init_dmatrix(double **d, long nr, long nc, double init_val);
double torsion(double **d);
double vec_ang(double *va, double *vb, double *vref);
void vec_orth(double *va, double *vref);
double dot(double *va, double *vb);
void cross(double *va, double *vb, double *vc);
double veclen(double *va);
void vec_norm(double *va);
double dot2ang(double dotval);
double magang(double *va, double *vb);
double rad2deg(double ang);
double deg2rad(double ang);
void get_BDIR(char *BDIR, char *filename);
void check_slash(char *BDIR);
void copy_matrix(double **a, long nr, long nc, double **o);
void multi_vec_matrix(double *a, long n, double **b, long nr, long nc, double *o);
void multi_matrix(double **a, long nra, long nca, double **b, long nrb, long ncb, double **o);
void multi_vec_matrix(double *a, long n, double **b, long nr, long nc, double *o);
void multi_vec_Tmatrix(double *a, long n, double **b, long nr, long nc, double *o);
void transpose_matrix(double **a, long nr, long nc, double **o);
void cov_matrix(double **a, double **b, long nr, long nc, double **cmtx);
void ls_fitting(double **sxyz, double **exyz, long n, double *rms_value,
                double **fitted_xyz, double **R, double *orgi);
void ls_plane(double **bxyz, long n, double *pnormal, double *ppos,
              double *odist, double *adist);
void identity_matrix(double **d, long n);
void arb_rotation(double *va, double ang_deg, double **rot_mtx);
void get_vector(double *va, double *vref, double deg_ang, double *vo);
void rotate(double **a, long i, long j, long k, long l,
            double *g, double *h, double s, double tau);
void eigsrt(double *d, double **v, long n);
void jacobi(double **a, long n, double *d, double **v);
void dludcmp(double **a, long n, long *indx, double *d);
void dlubksb(double **a, long n, long *indx, double *b);
void dinverse(double **a, long n, double **y);
long get_round(double d);
void fig_title(FILE * fp);
void ps_title_cmds(FILE * fp, char *imgfile, long *bbox);
void get_ps_xy(char *imgfile, long *urxy, long frame_box, FILE * fp);
void bring_atoms(long ib, long ie, long rnum, char **AtomName, long *nmatch, long *batom);
void all_bring_atoms(long num_residue, long *RY, long **seidx,
                     char **AtomName, long *num_ring, long **ring_atom);
void base_idx(long num, char *bseq, long *ibase, long single);
void plane_xyz(long num, double **xyz, double *ppos, double *nml, double **nxyz);
void prj2plane(long num, long rnum, char **AtomName, double **xyz, double z0, double **nxyz);
void adjust_xy(long num, double **xyz, long nO, double **oxyz,
               double scale_factor, long default_size, long *urxy);
void check_Watson_Crick(long num_bp, char **bp_seq, double **orien,
                        double **org, long *WC_info);
void base_frame(long num_residue, char *bseq, long **seidx, long *RY,
                char **AtomName, char **ResName, char *ChainID,
                long *ResSeq, char **Miscs, double **xyz, char *BDIR,
                double **orien, double **org);
void baseinfo(char chain_id, long res_seq, char iCode, char *rname,
              char bcode, long stnd, char *idmsg);
void hb_numlist(long i, long j, long **seidx, char **AtomName,
     double **xyz, double *HB_UPPER, char *HB_ATOM, long *num_hb, long **num_list);
void get_hbond_ij(long i, long j, double *HB_UPPER, long **seidx,
                  char **AtomName, char *HB_ATOM, double **xyz, 
                  long *nh, char **hb_atm1, char **hb_atm2, double *hb_dist); 
void hb_crt_alt(double *HB_UPPER, char *HB_ATOM, char *ALT_LIST);
void atom_info(long idx, char atoms_list[NELE][3], double *covalence_radii, double *vdw_radii);
void atom_idx(long num, char **AtomName, long *idx);
void atom_linkage(long ib, long ie, long *idx, double **xyz,
                  long nbond_estimated, long *nbond, long **linkage);
void del_extension(char *pdbfile, char *parfile);
void check_pair(long i, long j, char *bseq, long **seidx, double **xyz,
                double **Nxyz, double **orien, double **org, char **AtomName,
                double *BPRS, double *rtn_val, long *bpid, long network, char *);
void o3_p_xyz(long ib, long ie, char *aname, char **AtomName, double **xyz,
              double *o3_or_p, long idx);
void base_info(long num_residue, char *bseq, long **seidx, long *RY,
               char **AtomName, char **ResName, char *ChainID,
               long *ResSeq, char **Miscs, double **xyz, double **orien,
               double **org, double **Nxyz, double **o3_p, double *BPRS);
void bpstep_par(double **rot1, double *org1, double **rot2, double *org2,
                double *pars, double **mst_orien, double *mst_org);
void contact_msg(void);

void protein_rna_interact(double H_limit,long num_residue, long **seidx,
                          double **xyz, char **AtomName, long *prot_rna);


/* find_pair.c */
void usage(void);
  void cmdline(int argc, char *argv[], char *inpfile);

void handle_str(char *pdbfile, char *outfile, long ds, long curves,
                long divide, long hetatm, long pairs);
void all_pairs(char *pdbfile, FILE *fout, long num_residue, long *RY,
               double **Nxyz, double **orien, double **org, double *BPRS,
               long **seidx, double **xyz, char **AtomName, char **ResName,
               char *ChainID, long *ResSeq, char **Miscs, char *bseq,
               long *num_pair_tot, char **pair_type,long **bs_pairs_tot,
               long *num_single_base, long *single_base,long *num_multi,
               long *multi_idx, long **multi, long *sugar_syn);

void best_pair(long i, long num_residue, long *RY, long **seidx,
               double **xyz, double **Nxyz, long *matched_idx,
               double **orien, double **org, char **AtomName, char *bseq,
               double *BPRS, long *pair_stat);
void bp_context(long num_bp, long **base_pairs, double HELIX_CHG,
                double **bp_xyz, long **bp_order, long **end_list,
                long *num_ends);
void locate_helix(long num_bp, long **bp_order, long **end_list, long num_ends,
                  long *num_helix, long **helix_idx, long *bp_idx,
                  long *helix_marker);
void get_ij(long m, long *swapped, long **base_pairs, long *n1, long *n2);
void get_d1_d2(long m, long n, long *swapped, double **bp_xyz, double *d1, double *d2);
void re_ordering(long num_bp, long **base_pairs, long *bp_idx,
                 long *helix_marker, long **helix_idx, double *BPRS,
                 long *num_helix, double **o3_p, char *bseq, long **seidx,
                 char **ResName, char *ChainID, long *ResSeq, char **Miscs);
void helix_info(long **helix_idx, long idx, FILE * fp);
void write_bestpairs(long num_bp, long **base_pairs, long *bp_idx,
                     char *bseq, long **seidx, char **AtomName,
                     char **ResName, char *ChainID, long *ResSeq,
                     char **Miscs, double **xyz);
void write_helix(long num_helix, long **helix_idx, long *bp_idx,
                 long **seidx, char **AtomName, char **ResName,
                 char *ChainID, long *ResSeq, char **Miscs, double **xyz,
                 long **base_pairs);


/* yang's new routine */
void edge_type(long nh, char **hb_atm, long i, char *bseq, char *type_wd);
void get_pair_type(long num_hbonds, char **hb_atom1, char **hb_atom2,
                   long i, long j, char *bseq, char *type);
void get_atom_xyz(long ib, long ie, char *aname, char **AtomName, 
		  double **xyz, double *atom_xyz);
void cis_or_trans(long i, long j, char *bseq, long **seidx, char **AtomName,
                  double **xyz, char *cis_tran);
void LW_pair_type(long i, long j, double dist, long **seidx, char **AtomName,
                  char *HB_ATOM, double **xyz,char *bseq, char **hb_atom1,
                  char **hb_atom2, double *hb_dist, char *type);


void NC_vector(long i,long ib, long ie, char **AtomName,char *bseq, 
               double **xyz1, double *N_xyz, double *xyz,  double *vector_NC);
void middle_xyz(long nh,long ib, long ie, char **hb_atm, char **AtomName, double **xyz, 
		double *xyz1);
void rot_2_Yaxis (long num_residue,  double *z, long **seidx, double **xyz );

void rot_mol (long num_residue,  char **AtomName,char **ResName, char *ChainID,
              long *ResSeq, double **xyz, long **seidx, long *RY);
void process_3d_fig();
/*
void process_2d_fig(long num_residue, char *bseq, long **seidx, long *RY,
                    char **AtomName, char **ResName, char *ChainID,
                    long *ResSeq,char **Miscs, double **xyz,
                    long num_pair_tot, char **pair_type, long **bs_pairs_tot,
                    long num_helixs, long **helix_idx, long *bp_idx, long **base_pairs,
                    double **xy_bs, long *num_loop1, long **loop);
*/
void process_2d_fig(long num_residue, char *bseq, long **seidx, long *RY,
                    char **AtomName, char **ResName, char *ChainID,
                    long *ResSeq,char **Miscs, double **xyz,
                    long num_pair_tot, char **pair_type, long **bs_pairs_tot,
                    long num_helixs, long **helix_idx, long *bp_idx,
                    long **base_pairs,double **xy_bs,long *num_loop1,long **loop,
                    long *xmlnh, long *xml_helix_len, long **xml_helix,
                    long *xml_ns, long *xml_bases);


void rot_2_lsplane(long num, char **AtomName,  double **xyz);
void helix_regions(long num_helixs,long **helix_idx,long *bp_idx,long **base_pairs,
                   long *nhelix, long *npair_per_helix, long **bs_1,long **bs_2);

void rest_bases(long num_residue, long nhelix, long *npair_per_helix, long **bs_1,
                long **bs_2, long *nsub, long *bs_sub);
void rest_pairs(long bs_pair_tot, long **ResNum_pair, long nhelix,
                long *npair_per_helix, long **bs_1, long **bs_2, 
                long *npsub,long **bsp_sub);


void head_to_tail(long j, long *npair_per_helix, long **bs_1, long **bs_2,
                  long *nregion,long **sub_helix1,long **sub_helix2);

void helix_head(long k, long n, long **bs_1, long **bs_2, long *nsub,
                long *bs_sub, char *ChainID,long **seidx,
                long *loop,long *yes);
void helix_tail(long k, long n, long **bs_1, long **bs_2, long *nsub,
                long *bs_sub, char *ChainID, long **seidx,
                long *loop, long *yes);

void loop_proc(long k, long n1, long n2,long *nsub,
               long *bs_sub, char *ChainID,long **seidx, long *yes);
void add_bs_2helix(long i,long j,long n1,long n2, long num_residue,
                   long **bs_1, long **bs_2,long *nsub,
                   long *bs_sub,long *add, long *bs_1_add, long *bs_2_add);
void check_link(long i,long n1,long n2, long nsub,long *bs_sub,long **bs_1,
                long **bs_2, long *yes);
long chck_lk(long diff, long m1, long m2, long nsub, long *bs_sub);


void check(long m, long *nsub,long *bs_sub, long *yes);
void new_xy(double a, double d1, double *xy1, double *xy2);

void gen_xy_cood(long i,long num_residue,long n, double a, double *xy1, double *xy2,
                 long **bs_1,long **bs_2,long *nsub, long *bs_sub,
                 long **loop,char *ChainID,long **seidx,
                 long *link, long **bs1_lk, long **bs2_lk, double **xy_bs);


void link_xy(long i, long j, double d, long *nsub, long *bs_sub,
             long **bs_1, long **bs_2, double **xy_bs, long *num);

void link_xy_proc(long m, double d, long k01, long k02, long k1,
                  long **bs_1, long **bs_2, double **xy_bs);
void link_helix(long n, long n1,long n2,long num_residue, double d1, double d2, 
                double a,char *ChainID,long **seidx,double *xy1, double *xy2,
                long *nsub, long *bs_sub, double **xy_bs);



void loop_xy(long i, long n, long **bs_1, long **bs_2, double *xy1,double *xy2, 
             double a, double **xy_bs);
void loop_xy_proc(long i, long n, long m, double alfa,double ang, double ap, long **bs_1, 
                  double r, double x0,double y0,double **xy_bs);
void dashline_red(FILE *psfile, double *xy1, double *xy2);


void xy4ps(long n,double **oxy, long num, double **nxy);

void ps_head(FILE *psfile, char *imgfile, long *bbox);

void shapes(FILE *psfile, char *bseq, long k1, long k2,char *pair_type, double *x, double *y, double at, double ratio);
void twopoints(double *xy0, double a, double d, double *xy1, double *xy2);

void line(FILE *psfile, double *xy1, double *xy2);
void square(FILE *psfile, long fill, double *xy1, double *xy2, double d, double a);
void circle(FILE *psfile, long fill, double *xy1, double *xy2, double r);
void triangle(FILE *psfile, long fill, double *xy1, double *xy3, double d, double a);


void write_best_pairs(long num_helix, long **helix_idx, 
                      long *bp_idx, long *helix_marker, long **base_pairs,
                      long **seidx, char **ResName, char *ChainID,
                      long *ResSeq, char **Miscs, char *bseq, double *BPRS,
                      long *num_bp, long **pair_num);
void linefit(double *x, double *y, long n, double *a, double *b);
void  xy_on_axis(double a, double b, long n, double *xa, double *ya,
                 double *xy1, double *xy2);
void ringcenter(long i_order, long **seidx, char **AtomName, double **xyz,
                double *x1, double *y1);
void dashline(FILE *psfile, double *xy1, double *xy2);
void color_line(FILE *psfile, char *color,double *xy1, double *xy2);

void dotline(FILE *psfile, double *xy1, double *xy2);
void helix_head_tail(long num_residue, long i, long j, long **bs_1, 
                     long **bs_2, long  *nsub, long *bs_1_sub,long *bs_2_sub, 
                     long *nbs1,long *nbs2, long **nbs_1, long **nbs_2);

void reduce(long n,  long *nsub, long *bs_1_sub, long *bs_2_sub);

void HelixAxis(long ii , long n, long **bs_1, long **bs_2, long num_residue,
               long **seidx,long *RY, double **xyz, char **AtomName,char *bseq,
               double *a, double *xy1, double *xy2, double *helix_len, double **);
void axis_start_end(long num_bp, long num, long **chi, double **xyz,
                    double *hstart, double *hend);
void get_chi(long i, long ii, long n, long **bs_1, long **seidx, char **AtomName,
             char *bseq, long *RY, long **chi);
void decrease(long num_residue, long i, long j, long *kk, long **bs_pair, long *nsub, 
              long *bs_1_sub, long *bs_2_sub, long *nbs_1, long *nbs_2);
void increase(long num_residue, long i, long j, long *kk, long **bs_pair, long *nsub, 
              long *bs_1_sub, long *bs_2_sub, long *nbs_1, long *nbs_2);


void xy_at_base12(double a0, double b0,double  a1, double b1, long j,
                  double xi,double yi, double xf, double yf, double *x12, double *y12);

void xy_base12(long j, long n,double a, double d1, double d2,
               double *xy1, double *xy2, double *x12, double *y12);

void xy_base(long j, long n, long n1, long n01, double a, double d1, 
             double *xy1, double *xy2, double **xy_bs);

void write_xml(char *parfile, long num_residue, char *bseq, long **seidx,
               char **AtomName, char **ResName, char *ChainID, long *ResSeq,
               char **Miscs, double **xyz, long xml_nh, long **xml_helix,
               long *xml_helix_len, long xml_ns, long *xml_bases,
               long num_pair_tot, long **bs_pairs_tot, char **pair_type,
               double **base_xy, long num_modify, long *modify_idx,
               long num_loop, long **loop, long num_multi,long *multi_idx,
               long **multi, long *sugar_syn);

void real_helix(long num_residue,long num_helixs,long **helix_idx,long *bp_idx,
                long **base_pairs, long *nhelix, long **bs_pair_start,
                long *helix_length);
void single_strand(long num_residue, long nhelix, long *npair_per_helix,
                   long **bs_1, long **bs_2,long *num_strand,long **sing_st_end);

void single_continue(long num_single_base, long *single_base, long *num_strand,
                     long **sing_st_end);


void re_ordering(long num_bp, long **base_pairs, long *bp_idx,
                 long *helix_marker, long **helix_idx, double *BPRS,
                 long *num_helix, double **o3_p, char *bseq, long **seidx,
                 char **ResName, char *ChainID, long *ResSeq, char **Miscs);

void bp_context1(long num_bp, long **base_pairs, double HELIX_CHG,
                double **bp_xyz, long **bp_order, long **end_list,
                long *num_ends);
void locate_helix1(long num_bp, long **helix_idx, long num_ends,
                  long *num_helix, long **end_list, long **bp_order,
                  long *bp_idx, long *helix_marker);

double distance_ab(double **o3_p, long ia, long ib, long ipa, long ipb);
void get_ij(long m, long *swapped, long **base_pairs, long *n1, long *n2);
void get_d1_d2(long m, long n, long *swapped, double **bp_xyz, double *d1, double *d2);
void check_zdna(long *num_helix, long **helix_idx, long *bp_idx,
                double **bp_xyz);

void five2three(long num_bp, long *num_helix, long **helix_idx,
                long *bp_idx, double **bp_xyz, long **base_pairs,
                double **o3_p);

void lreverse(long ia, long n, long *lvec);

void lsort(long n, long *a, long *idx);

void bp_network(long num_residue, long *RY, long **seidx, char **AtomName,
                char **ResName, char *ChainID, long *ResSeq, char **Miscs,
                double **xyz, char *bseq, long **pair_info, double **Nxyz,
                double **orien, double **org, double *BPRS, FILE * fp,
               long *num_multi, long *multi_idx, long **multi);

void multiplets(long num_ple, long max_ple, long num_residue,
                long **pair_info, long *ivec, long *idx1, char **AtomName,
                char **ResName, char *ChainID, long *ResSeq, char **Miscs,
                double **xyz, double **orien, double **org, long **seidx,
                char *bseq, FILE * fp,
               long *num_multi, long *multi_idx, long **multi);
