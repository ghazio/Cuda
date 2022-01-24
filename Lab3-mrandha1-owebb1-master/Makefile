C = gcc
C++ = g++
CUDA = /usr/local/cuda/bin/nvcc
CFLAGS = -g -Wall -Wvla -Werror -Wno-error=unused-variable


# TODO this needs to be set for each card type
#CUDA architecture. Use deviceQuery |grep Major to find out version 
#ARCH = 75
# thyme is version number 6.1
#ARCH = 61
# our oldest graphics card (2021):
ARCH = 50

#cuda include/library paths
CUDAINCDIR = -I/usr/local/cuda-11.2/include
CUDALIBDIR = -L/usr/local/cuda-11.2/lib64

#qtvis include path
INCLUDEDIR = -I/usr/local/include/qtvis

#qt5 include path
QTINCDIR = -I/usr/include/x86_64-linux-gnu/qt5

#CUDA libraries, options
CUDALIBS = -lqtviscuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl
COMPUTE = -forward-unknown-to-host-compiler \
          --generate-code=arch=compute_$(ARCH),code=[compute_$(ARCH),sm_$(ARCH)]

#Qt5 stuff
QTINCLUDES = $(QTINCDIR) \
	     $(QTINCDIR)/QtOpenGL \
	     $(QTINCDIR)/QtWidgets \
	     $(QTINCDIR)/QtGui \
	     $(QTINCDIR)/QtCore
DEFINES = -DQT_CORE_LIB -DQT_GUI_LIB -DQT_OPENGL_LIB -DQT_WIDGETS_LIB
OPTIONS = -fPIC
LIBS = $(LIBDIR) -lqtvis -lQt5OpenGL -lQt5Widgets -lQt5Gui -lQt5Core -lGLX -lOpenGL -lpthread

MAINPROG=main
CUDAPROG=firesimulator

all: $(CUDAPROG)

#linking with link path and libs
$(CUDAPROG): $(MAINPROG).o $(CUDAPROG).o
	$(C++) -g $(CUDALIBDIR) -o $(CUDAPROG) \
	   $(MAINPROG).o $(CUDAPROG).o $(LIBS) $(CUDALIBS)

#build the Qt5 side with no CUDA code/compiler
# this is to compile qt code that does not have CUDA 
# (does need to include cudadir to compile), but can't include cuda
# kernel or device code in this, but can include cuda.h
$(MAINPROG).o: $(MAINPROG).cpp
	$(C++) $(CFLAGS) $(CUDAINCDIR) $(QTINCLUDES) $(INCLUDEDIR)\
		$(OPTIONS) -c $(MAINPROG).cpp

#build the cuda specific part with nvcc
# this is to compile cuda code that does not qt
$(CUDAPROG).o: $(CUDAPROG).cu $(CUDAPROG).h
	$(CUDA) -g $(INCLUDEDIR) $(COMPUTE) -o $(CUDAPROG).o -c $(CUDAPROG).cu

clean:
	$(RM) $(CUDAPROG) *.o

