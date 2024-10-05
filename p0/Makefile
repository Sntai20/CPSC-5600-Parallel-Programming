
all: out p0 example

out:
	mkdir out

p0: p0.cpp Makefile
	g++ p0.cpp -o out/p0 -fopenmp

example: example.cpp Makefile
	g++ example.cpp -o out/example -fopenmp