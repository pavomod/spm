CXX      = g++
CXXFLAGS = -O3 -std=c++20 -Wall -Wextra -fopenmp -I ../common

.PHONY: all clean

all: omp_ws

omp_ws: iterative_SpMV_omp_ws.cpp ../common/matrix_generation.hpp ../common/utils.hpp
	$(CXX) $(CXXFLAGS) iterative_SpMV_omp_ws.cpp -o omp_ws

clean:
	rm -f omp_ws
