/* General purpose header file - rf 12/90 */

#ifndef _H_general
#define _H_general



/* Macintosh specific */
#ifdef MAC					/* rf 12/9/94 */

#define const					/* THINK C doesn't know about these identifiers */
#define signed
#define volatile
#define int long
#ifndef Boolean
#define Boolean char
#endif
#define pint short			/* cast ints in printf statements as pint */
#define sint int			/* cast ints for sequence lengths */
#define lint int			/* cast ints for profile scores */

#else 							/* not Macintoshs */

#define pint int			/* cast ints in printf statements as pint */
#define sint int			/* cast ints for sequence lengths */
#define lint int 			/* cast ints for profile scores */
#ifndef Boolean
#define Boolean char
#endif

#endif 							/* ifdef MAC */

/* definitions for all machines */

#undef TRUE						/* Boolean values; first undef them, just in case */
#undef FALSE
#define TRUE 1
#define FALSE 0

#define EOS '\0'				/* End-Of-String */
#define MAXLINE 512			/* Max. line length */


#ifdef VMS
#define signed
#endif


#endif /* ifndef _H_general */

