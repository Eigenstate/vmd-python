#ifndef PSFTYPELIST
#define PSFTYPELIST

class PsfTypeList {
	public:
		PsfTypeList() {
			head=NULL;
		}

		~PsfTypeList() {
			delete head;
		}

		bool search(PsfType*);

		void addtype(PsfType*);

		PsfType* lookForType(PsfType*);

		void print(FILE*);

	private:
		PsfType* head;
};

#endif
