CXX      = g++
CXXFLAGS = -O3 -std=c++20 -Wall -Wextra -pthread -I ../common

.PHONY: all clean

all: threads

threads: iterative_SpMV_threads.cpp ../common/matrix_generation.hpp ../common/utils.hpp
	$(CXX) $(CXXFLAGS) iterative_SpMV_threads.cpp -o threads

clean:
	rm -f threads
