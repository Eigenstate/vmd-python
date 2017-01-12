#ifndef PSFRESLIST
#define PSFRESLIST

class PsfResList {
	public:
		PsfResList() {
			head=NULL;
			tail=NULL;
		}

		~PsfResList() {
			delete head;
		}

		bool search(PsfAtom*);

		void addres(PsfRes*);

		PsfRes* lookForRes(PsfAtom*);

		void print(FILE*);

	private:
		PsfRes* head;
		PsfRes* tail;
};

#endif
