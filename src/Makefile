
mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(dir $(mkfile_path))
DESTDIR := $(abspath $(current_dir)../bin)
DATA_SRCDIR=../data

all:
	@echo "Run 'make install'"

${DESTDIR}/mavelp/__init__.py: mavelp/__init__.py mavelp/data_tool.py mavelp/gui.py mavelp/kernel_methods.py mavelp/neural_network.py
	cd .. && pip install -t ${DESTDIR} .

${DESTDIR}/data/data.dat: ${DATA_SRCDIR}/data.dat
	cp -r ../data ${DESTDIR}
	
mavelp: ${DESTDIR}/mavelp/__init__.py ${DESTDIR}/data/data.dat

install:
	@echo "Installing to: ${DESTDIR}"
	. /etc/environ.sh; use -e -r anaconda3-5.1; $(MAKE) mavelp

clean:
	rm -rf ${DESTDIR}/*.dist-info

distclean: clean
	rm -rf ${DESTDIR}/mavelp
	rm -rf ${DESTDIR}/bin
	rm -rf ${DESTDIR}/data

.PHONY: all install clean distclean
