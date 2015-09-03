#ifndef _PSFATOM
#define _PSFATOM

class PsfAtom {
	public:
		PsfAtom(char*,char*); 
		PsfAtom(char*, char*,char*,char*,char*,char*); 

		bool equals(PsfAtom);

		void setnext(PsfAtom*);

		bool search(PsfAtom*);

		PsfAtom* find_index(int);

		int index();

		int resid();

		char* name();

		char* res();

		char* elem();

		void print(FILE*);

		void alias();

		void aliasnucleic();

    PsfAtom* getnext();

	private:
		int  pdbindex;
		int  residueid;
		char atomname[6];
		char resname[5];
		char segname[5];
		char element[3];
		PsfAtom* next;
};

char *eatwhite(char *s);
char *trimright(char *s, int);

#endif
