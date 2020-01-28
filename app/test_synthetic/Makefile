SHELL = /bin/sh
ARCH = macg
TARGET = arma_model
CSRCS = arma_model.cpp
OBJS = $(CSRCS:.c=.o)
DEPSRC = $(SRCS)
DEPCSRC = $(CSRCS)
LIBS = -L$(KARMA4_PATH)/lib -L$(NLOPT_PATH)/lib -lkarma4 -lnlopt
INCL = -I$(KARMA4_PATH)/include -I$(NLOPT_PATH)/include
CPP = g++
CPPFLAGS = -O2 $(INCL)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CPP) $(CPPFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

clean:
	rm -f *.o $(TARGET)


