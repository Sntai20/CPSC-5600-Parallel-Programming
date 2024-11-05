all: out p4

out:
	mkdir out

# p4: p4.cpp Makefile
# 	g++ p4.cpp -o out/p4 -lpthread -Wall -Werror --std=c++20

# p4: p4.cpp Makefile
# 	g++ p4.cpp -o out/p4 -lpthread -Wall -Werror --std=c++20 -DCUSTOM_SHAREDPOINTER

# p4: p4.cpp Makefile
# 	g++ p4.cpp -o out/p4 -lpthread -Wall -Werror --std=c++20 -DCUSTOM_ATOMICITERATOR

p4: p4.cpp Makefile
	g++ p4.cpp -o out/p4 -lpthread -Wall -Werror --std=c++20 -DCUSTOM_SHAREDPOINTER -DCUSTOM_ATOMICITERATOR