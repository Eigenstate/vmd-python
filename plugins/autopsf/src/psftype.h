#ifndef _PSFTYPE
#define _PSFTYPE

class PsfType {
	public:
		PsfType(char*); 

		PsfType(char*, float, char*); 

		~PsfType(); 

		bool equals(PsfType);

		void setnext(PsfType*);

		bool search(PsfType*);

		char* type();

		float mass();

		char* elem();

		void print(FILE*);

	private:
		char  atomtype[10];
		int   index;
		float atommass;
		char  element[10];
		PsfType* next;
};

char *eatwhite(char *s);

#endif
