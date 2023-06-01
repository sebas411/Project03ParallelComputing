all: pgm.o	hough houghConstant houghShared

hough:	hough.cu pgm.o
	nvcc hough.cu pgm.o -o hough -ljpeg

houghConstant:	houghConstant.cu pgm.o
	nvcc houghConstant.cu pgm.o -o houghConstant -ljpeg

houghShared:	houghShared.cu pgm.o
	nvcc houghShared.cu pgm.o -o houghShared -ljpeg

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o -ljpeg

clean:
	-rm *.o hough houghConstant
