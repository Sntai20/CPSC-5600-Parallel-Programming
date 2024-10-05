
all: out p1 example

out:
	mkdir out

p1: p1.cpp Makefile
	g++ p1.cpp -lpthread -o out/p1

example: example.cpp Makefile
	g++ example.cpp -lpthread -o out/example