run.o:main.cpp Matrix.h UMatrix.h
	g++ main.cpp Matrix.h UMatrix.h -fopenmp -O2
