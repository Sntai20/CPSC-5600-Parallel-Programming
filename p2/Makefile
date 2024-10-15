# Copyright (c) 2024 Braxton Cuneo, under MIT License

all: out p2

out:
	mkdir out

p2: p2.cpp grid.h grid.cpp input.h input.cpp input_set.h
	g++ --std=c++20 p2.cpp grid.cpp input.cpp -o out/p2 -lpthread -g