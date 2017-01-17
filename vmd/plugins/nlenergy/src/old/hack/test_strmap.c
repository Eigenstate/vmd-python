/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * test_strmap.c - Demonstrate use of Strmap and getopt()
 */

#define VERSION "1.00"

/*
 * define _GNU_SOURCE macro to use GNU extension getopt_long()
 * (might also need to link using -lbsd-compat)
 */

#include <unistd.h>
#ifdef _GNU_SOURCE
#include <getopt.h>
#endif
#include "nlbase/nlbase.h"

#define D(s)  NL_fprintf(stderr, "DEBUG: %s\n", s);
#define X(n)  NL_fprintf(stderr, "DEBUG: %s=%x\n", #n, n);

#define NELEMS(x)  (sizeof(x) / sizeof(x[0]))

struct month_t {
  const char *name;
  const char *fullname;
  int number;
  int days;
  int leapdays;
};

static const struct month_t Month[] = {
  { "jan", "January",   1, 31, 31 },
  { "feb", "February",  2, 28, 29 },
  { "mar", "March",     3, 31, 31 },
  { "apr", "April",     4, 30, 30 },
  { "may", "May",       5, 31, 31 },
  { "jun", "June",      6, 30, 30 },
  { "jul", "July",      7, 31, 31 },
  { "aug", "August",    8, 31, 31 },
  { "sep", "September", 9, 30, 30 },
  { "oct", "October",  10, 31, 31 },
  { "nov", "November", 11, 30, 30 },
  { "dec", "December", 12, 31, 31 }
};

#define OPT_FULLNAME  0x0001
#define OPT_NUMBER    0x0002
#define OPT_DAYS      0x0004
#define OPT_LEAPDAYS  0x0008
#define OPT_ALL       (OPT_FULLNAME | OPT_NUMBER | OPT_DAYS | OPT_LEAPDAYS)
#define OPT_VERSION   0x0010
#define OPT_STATS     0x0020
#define OPT_HELP      0x0040

#define OPT_MULTI_MONTHS     0x0100
#define OPT_FULLNAME_COMMA   0x0200
#define OPT_NUMBER_PREFIX    0x0400
#define OPT_NUMBER_COMMA     0x0800
#define OPT_DAYS_SUFFIX      0x1000
#define OPT_DAYS_COMMA       0x2000
#define OPT_LEAPDAYS_SUFFIX  0x4000

#ifdef _GNU_SOURCE
static const struct option Option[] = {
  { "help", 0, NULL, 'h' },
  { "version", 0, NULL, 'v' },
  { "fullname", 0, NULL, 'f' },
  { "number", 0, NULL, 'n' },
  { "days", 0, NULL, 'd' },
  { "leapdays", 0, NULL, 'l' },
  { "all", 0, NULL, 'a' },
  { "stats", 0, NULL, 's' },
  { "output", 1, NULL, 'o' }
};
#endif

static void help(FILE *out, const char *fname)
{
  NL_fprintf(out,
"Obtain name and information about each month\n"
"(actual purpose is to demonstrate use of getopt() and Strmap)\n"
"\n"
      );
  NL_fprintf(out,
"Usage: %s [OPTION] MONTH [ MONTH ... ]\n"
"\n"
      , fname);
  NL_fprintf(out,
"  MONTH is the three letter abbreviation for a month\n"
"\n"
"  OPTION list:\n"
"\n"
      );

#ifdef _GNU_SOURCE
  NL_fprintf(out,
"  -h, --help            display this help and exit\n"
"  -v, --version         print version information and exit\n"
"\n"
      );
  NL_fprintf(out,
"  -f, --fullname        print full name of month (default)\n"
"  -n, --number          print number of month\n"
"  -d, --days            print number of days in month\n"
"  -l, --leapdays        print number of days in month during leap year\n"
"  -a, --all             all of -f, -n, -d, -l options together\n"
"\n"
      );
  NL_fprintf(out,
"  -s, --stats           print hash table statistics\n"
"\n"
"  -o FILE, --output FILE        save output to FILE\n"
      );

#else
  NL_fprintf(out,
"  -h            display this help and exit\n"
"  -v            print version information and exit\n"
"\n"
      );
  NL_fprintf(out,
"  -f            print full name of month (default)\n"
"  -n            print number of month\n"
"  -d            print number of days in month\n"
"  -l            print number of days in month during leap year\n"
"  -a            all of -f, -n, -d, -l options together\n"
"\n"
      );
  NL_fprintf(out,
"  -s            print hash table statistics\n"
"\n"
"  -o FILE       save output to FILE\n"
      );

#endif
  exit(1);
}

static void version(FILE *out, const char *fname)
{
  NL_fprintf(out,
"%s - Version %s\n"
"\n"
"Obtain name and information about each month\n"
"(actual purpose is to demonstrate use of getopt() and Strmap)\n"
"\n"
"Copyright (C) 2007 by David J. Hardy.  All rights reserved.\n"
    , fname, VERSION);
  exit(1);
}

int main(int argc, char **argv)
{
  extern char *optarg;
  extern int optind, opterr, optopt;
  int c;
  int option = 0;
  const char *fname = NULL;
  FILE *out;
  Strmap strmap;
  Strmap *t = &strmap;
  int k;

  while (
#ifdef _GNU_SOURCE
      (c = getopt_long(argc, argv, "afndlo:hvs", Option, NULL))
#else
      (c = getopt(argc, argv, "afndlo:hvs"))
#endif
      != EOF) {
    switch (c) {
      case 'a':
        option |= OPT_ALL;
        break;
      case 'f':
        option |= OPT_FULLNAME;
        break;
      case 'n':
        option |= OPT_NUMBER;
        break;
      case 'd':
        option |= OPT_DAYS;
        break;
      case 'l':
        option |= OPT_LEAPDAYS;
        break;
      case 'o':
        fname = optarg;
        break;
      case 'v':
        option |= OPT_VERSION;
        break;
      case 's':
        option |= OPT_STATS;
        break;
      case 'h':
      case '?':
        option |= OPT_HELP;
        break;
      default:
        NL_fprintf(stderr, "error: default target reached!\n");
        exit(1);
    }
  }

  if ((option & OPT_HELP) || (optind == argc && !(option & OPT_VERSION))) {
    help(stdout, argv[0]);
  }
  else if (option & OPT_VERSION) {
    version(stdout, argv[0]);
  }

  if (option == 0) {
    option |= OPT_FULLNAME;
  }

  if (fname) {
    if ((out = fopen(fname, "w")) == NULL) {
      NL_fprintf(stderr, "%s: failed to open file %s for writing\n",
          argv[0], fname);
      exit(1);
    }
  }
  else {
    out = stdout;
  }

  if (Strmap_init(t,0)) {
    NL_fprintf(stderr, "%s: failed call to Strmap_init()\n", argv[0]);
    exit(1);
  }
  else if (option & OPT_STATS) {
    NL_fprintf(out, "hash table: %s\n", Strmap_stats(t));
  }

  if (option & OPT_STATS) {
    NL_fprintf(out, "(inserting abbreviations for 12 months)\n");
  }
  for (k = 0;  k < NELEMS(Month);  k++) {
    if (Strmap_insert(t, Month[k].name, k) != k) {
      NL_fprintf(stderr, "%s: failed call to insert() for \"%s\"\n",
          argv[0], Month[k].name);
      exit(1);
    }
  }
  if (option & OPT_STATS) {
    NL_fprintf(out, "hash table: %s\n", Strmap_stats(t));
  }

  if (argc - optind > 1) {
    option |= OPT_MULTI_MONTHS;
  }
  if ((option & OPT_FULLNAME)
      && (option & (OPT_NUMBER | OPT_DAYS | OPT_LEAPDAYS))) {
    option |= OPT_FULLNAME_COMMA;
  }
  if ((option & OPT_NUMBER)
      && (option & (OPT_FULLNAME | OPT_DAYS | OPT_LEAPDAYS))) {
    option |= OPT_NUMBER_PREFIX;
  }
  if ((option & OPT_NUMBER)
      && (option & (OPT_DAYS | OPT_LEAPDAYS))) {
    option |= OPT_NUMBER_COMMA;
  }
  if ((option & OPT_DAYS)
      && (option & (OPT_FULLNAME | OPT_NUMBER | OPT_LEAPDAYS))) {
    option |= OPT_DAYS_SUFFIX;
  }
  if ((option & OPT_DAYS)
      && (option & OPT_LEAPDAYS)) {
    option |= OPT_DAYS_COMMA;
  }
  if ((option & OPT_LEAPDAYS)
      && (option & (OPT_FULLNAME | OPT_NUMBER | OPT_DAYS))) {
    option |= OPT_LEAPDAYS_SUFFIX;
  }

  for ( ;  optind < argc ;  optind++) {
    if ((k = Strmap_lookup(t, argv[optind])) == FAIL) {
      NL_fprintf(stderr, "%s: \"%s\" is not the abbreviation for a month\n",
          argv[0], argv[optind]);
      option |= OPT_HELP;
    }
    else {
      if (option & OPT_MULTI_MONTHS) {
        NL_fprintf(out, "%s: ", Month[k].name);
      }
      if (option & OPT_FULLNAME) {
        NL_fprintf(out, "%s", Month[k].fullname);
      }
      if (option & OPT_FULLNAME_COMMA) {
        NL_fprintf(out, ", ");
      }
      if (option & OPT_NUMBER_PREFIX) {
        NL_fprintf(out, "month ");
      }
      if (option & OPT_NUMBER) {
        NL_fprintf(out, "%d", Month[k].number);
      }
      if (option & OPT_NUMBER_COMMA) {
        NL_fprintf(out, ", ");
      }
      if (option & OPT_DAYS) {
        NL_fprintf(out, "%d", Month[k].days);
      }
      if (option & OPT_DAYS_SUFFIX) {
        NL_fprintf(out, " days");
      }
      if (option & OPT_DAYS_COMMA) {
        NL_fprintf(out, ", ");
      }
      if (option & OPT_LEAPDAYS) {
        NL_fprintf(out, "%d", Month[k].leapdays);
      }
      if (option & OPT_LEAPDAYS_SUFFIX) {
        NL_fprintf(out, " days during leap year");
      }
      NL_fprintf(out, "\n");
    }
  }

  if (option & OPT_HELP) {
    NL_fprintf(out, "\n");
    help(out, argv[0]);
  }

  Strmap_done(t);
  return 0;
}
