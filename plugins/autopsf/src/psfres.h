#ifndef PSFRES
#define PSFRES
#include <string.h>
#include "psfbond.h"

class PsfRes {
	private:
		PsfAtom* head;
		PsfBond* bondhead;
		PsfRes* next;
		char resname[10];
		int patch;

	public:
		PsfRes(char* inres) {
			//strncpy(resname, inres, strlen(inres));
			strncpy(resname, inres, 9);
			if (strlen(inres) < 9) {
				resname[strlen(inres)]='\0';
			} else {
				resname[9]='\0';
			}
			head=NULL;
			bondhead=NULL;
			next=NULL;
			//fprintf(stdout,"New residue created: %s \n",resname);
		}

		~PsfRes() {
			while (head != NULL) {
        PsfAtom* temphead = head;
        head = head->getnext();
				delete temphead;
			}
			if (bondhead != NULL) {
				delete bondhead;
			}

			if (next != NULL) { 
				delete next;
			}
		}

		void print(FILE* outfile);

		int ispatch();

		char* name();

		PsfRes* checkRes(PsfAtom* lookatom);

		bool search(PsfAtom* lookatom);

		bool searchbond(PsfBond* lookbond);

		void addatom(PsfAtom* newatom);

		void addbond(PsfBond* newbond);

		void setnext(PsfRes* nextres);

};
#endif
