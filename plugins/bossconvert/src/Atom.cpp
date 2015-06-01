#include "Atom.h"
#include <cstdio>
using namespace std;

//bool contains(list<Bond*>, Atom *);



Atom::Atom(string n, string typCH, string typOP, float s, float e){
	
}

Atom::Atom(){
	
}

bool Atom::operator< (const Atom& other) const{
	return this->number < other.number;
}

bool Atom::operator== (const Atom& other) const{
	return this->number == other.number;
}

Atom::~Atom(){

}

/*void Atom::addBond(Atom *a, float f, float d){
	if(!contains(this->bondedTo, a)){
		Bond b;
		b.a1 = this;
		b.a2 = a;
		b.force = f;
		b.distance = d;
		this->bondedTo.push_back(&b);
	}
}*/

/*bool contains(list<Bond*> lst, Atom *item){
	list<Bond*>::iterator it;
	for(it = lst.begin(); it != lst.end(); it++){
		Bond b = **it;
		if(b.a1->name.compare(item->name) == 0 || b.a2->name.compare(item->name) == 0)
			return true;
	}
	return false;
}*/
