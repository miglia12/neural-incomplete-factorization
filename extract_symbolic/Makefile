UNAME := $(shell uname -s)

ifeq ($(UNAME),Darwin)
  CC       = gcc-13
  CFLAGS   = -O2 -g -fopenmp
  RTLIBS   = -lgfortran -lm
  LIBNAME  = libpardiso.dylib
else
  LIBNAME  = libpardiso.so
  ifneq ($(shell command -v icx 2>/dev/null),)
    CC      = icx
    CFLAGS  = -O3 -march=icelake-server -qopenmp
    RTLIBS  = -lm
  else
    CC      = gcc
    CFLAGS  = -O3 -march=icelake-server -fopenmp -funroll-loops
    RTLIBS  = -lgfortran -lm
  endif
endif

TARGET = extract_symbolic

$(TARGET): extract_symbolic.c pardiso.h mmio_csr.h $(LIBNAME)
	$(CC) $(CFLAGS) extract_symbolic.c -I. ./$(LIBNAME) -Wl,-rpath,'$$ORIGIN' $(RTLIBS) -o $@

clean:
	rm -f $(TARGET)

.PHONY: clean
