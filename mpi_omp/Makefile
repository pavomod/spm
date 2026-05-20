MPICXX   = mpicxx
CXXFLAGS = -O3 -std=c++20 -Wall -Wextra -fopenmp -I ../common

.PHONY: all clean

all: mpi_omp

mpi_omp: iterative_SpMV_mpi.cpp ../common/matrix_generation.hpp ../common/utils.hpp
	$(MPICXX) $(CXXFLAGS) iterative_SpMV_mpi.cpp -o mpi_omp

clean:
	rm -f mpi_omp
