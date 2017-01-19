#ifndef _PSFBOND
#define _PSFBOND

#include "psfatomlist.h"


class PsfBond {
	public:
		PsfBond(int); 

		PsfBond(char*, char*); 

		~PsfBond(); 

		bool equals(PsfBond);

		void setnext(PsfBond*);

		bool search(PsfBond*);

		char* name0();
		char* name1();

		void print(FILE*);

	private:
		char atomname0[5];
		char atomname1[5];
		PsfBond* next;
};

char *eatwhite(char *s);

#endif
