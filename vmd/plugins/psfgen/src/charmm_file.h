
#ifndef CHARMM_FILE_H
#define CHARMM_FILE_H

#include <stdio.h>

int charmm_get_tokens(char **tok, int toklen,
			char *sbuf, int sbuflen,
			char *lbuf, int *lineno,
			FILE *stream, int all_caps);

#endif

