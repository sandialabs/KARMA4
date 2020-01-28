SHELL = /bin/sh
TARGET = libkarma4.a
CSRCS = karma4.cpp
DEPCSRC = $(CSRCS)
OBJS = $(CSRCS:.cpp=.o)
LIBS = -L$(NLOPT_PATH)/lib -lnlopt
INCFILES = karma4.hpp
INCDIR=../include
INCL = -I. -I$(NLOPT_PATH)/include
CPP = g++
CPPFLAGS = -O2 -std=gnu++11 $(INCL)
RANLIB = ranlib
AR = ar cr

all: $(TARGET)

$(TARGET): $(OBJS)
	$(AR) $(TARGET) $(OBJS)
	$(RANLIB) $(TARGET)
	for i in $(INCFILES) ;\
	  do /bin/rm -f $(INCDIR)/$${i} ;\
	  /bin/cp $$i $(INCDIR)/$${i} ;\
	done

clean:
	rm -f $(OBJS) $(TARGET)

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $*.cpp $(LIBS)
