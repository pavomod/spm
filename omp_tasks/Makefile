CXX      = g++
CXXFLAGS = -O3 -std=c++20 -Wall -Wextra -fopenmp -I ../common

.PHONY: all clean

all: omp_tasks

omp_tasks: iterative_SpMV_omp_tasks.cpp ../common/matrix_generation.hpp ../common/utils.hpp
	$(CXX) $(CXXFLAGS) iterative_SpMV_omp_tasks.cpp -o omp_tasks

clean:
	rm -f omp_tasks
