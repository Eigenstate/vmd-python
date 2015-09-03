CXX=g++
CXXFLAGS=  -Wall

all: Topology

Topology: Parser.o Worker.o Atom.o Main.o
        $(CXX) $(CXXFLAGS) Parser.o Worker.o Atom.o Main.o -o Topology

Parser.o: Parser.cpp
        $(CXX) $(CXXFLAGS) -c Parser.cpp

Worker.o: Worker.cpp
        $(CXX) $(CXXFLAGS) -c Worker.cpp

Atom.o: Atom.cpp
        $(CXX) $(CXXFLAGS) -c Atom.cpp

Main.o: Main.cpp

clean:
	rm -rf *o Topology
