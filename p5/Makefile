mpixx = $(shell mpic++ --showme)

all: out p5

out:
	mkdir out

p5: p5.cpp cluster.cpp cluster.h Makefile
	$(mpixx) p5.cpp cluster.cpp -o out/p5 -lpthread -Wall -Werror --std=c++20 -g