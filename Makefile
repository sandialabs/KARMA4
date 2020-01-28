SHELL=/bin/sh

SUBDIRS=lib app

default: what

what:
	@echo "Specify a target."

all clean:
	@for i in $(SUBDIRS) ;\
	 do echo "Making $@ in $$i ..."; cd $$i; $(MAKE) $@; cd .. ;\
	 done
