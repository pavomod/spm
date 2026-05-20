CXX      = g++
CXXFLAGS = -O3 -std=c++20 -Wall -Wextra -I ../common

.PHONY: all clean

all: seq

seq: iterative_SpMV.cpp ../common/matrix_generation.hpp ../common/utils.hpp
	$(CXX) $(CXXFLAGS) iterative_SpMV.cpp -o seq

clean:
	rm -f seq
