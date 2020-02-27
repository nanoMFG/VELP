
mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(dir $(mkfile_path))
DESTDIR := $(abspath $(current_dir)../bin)

all:
	@echo "Run 'make install'"

mavelp:
	@echo ${DESTDIR}
	cd .. && pip install -t ${DESTDIR} .

install:
	. /etc/environ.sh; use -e -r anaconda3-5.1; $(MAKE) mavelp

clean:
	rm -rf ${DESTDIR}/*.dist-info

distclean: clean
	rm -rf ${DESTDIR}/mavelp
	rm -rf ${DESTDIR}/bin

.PHONY: all mavelp install clean distclean
