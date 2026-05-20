.PHONY: all seq threads omp_tasks omp_ws mpi_omp clean

all: seq threads omp_tasks omp_ws mpi_omp

seq:
	$(MAKE) -C seq

threads:
	$(MAKE) -C threads

omp_tasks:
	$(MAKE) -C omp_tasks

omp_ws:
	$(MAKE) -C omp_ws

mpi_omp:
	$(MAKE) -C mpi_omp

clean:
	$(MAKE) -C seq clean
	$(MAKE) -C threads clean
	$(MAKE) -C omp_tasks clean
	$(MAKE) -C omp_ws clean
	$(MAKE) -C mpi_omp clean
