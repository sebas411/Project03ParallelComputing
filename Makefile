all: pgm.o	hough houghConstant

hough:	hough.cu pgm.o
	nvcc hough.cu pgm.o -o hough -ljpeg

houghConstant:	houghConstant.cu pgm.o
	nvcc houghConstant.cu pgm.o -o houghConstant -ljpeg

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o -ljpeg

clean:
	-rm *.o hough houghConstant
