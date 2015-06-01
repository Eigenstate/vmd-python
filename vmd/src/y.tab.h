
typedef union
#ifdef __cplusplus
	YYSTYPE
#endif
 {
	int ival;
	double dval;
	atomparser_node *node;
} YYSTYPE;
extern YYSTYPE yylval;
# define KEY 257
# define WITHIN 258
# define EXWITHIN 259
# define PBWITHIN 260
# define WITHINBONDS 261
# define MAXRINGSIZE 262
# define RINGSIZE 263
# define WHERE 264
# define FUNC 265
# define STRFCTN 266
# define SAME 267
# define NEAREST 268
# define SINGLE 269
# define FROM 270
# define OF 271
# define AS 272
# define THROUGH 273
# define PARSEERROR 274
# define RANGE 275
# define FLOATVAL 276
# define INTVAL 277
# define STRWORD 278
# define COMPARE 279
# define OR 280
# define AND 281
# define LT 282
# define LE 283
# define EQ 284
# define GE 285
# define GT 286
# define NE 287
# define NLT 288
# define NLE 289
# define NEQ 290
# define NGE 291
# define NGT 292
# define NNE 293
# define SLT 294
# define SLE 295
# define SEQ 296
# define SGE 297
# define SGT 298
# define SNE 299
# define MATCH 300
# define ADD 301
# define SUB 302
# define MULT 303
# define DIV 304
# define MOD 305
# define EXP 306
# define nonassoc 307
# define NOT 308
# define UMINUS 309
