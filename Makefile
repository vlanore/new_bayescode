# ==================================================================================================
#  COMPILATION
# ==================================================================================================
.PHONY: all # Requires: cmake 3.1.0 or better
all: _build
	@cd _build ; make --no-print-directory -j8

_build: CMakeLists.txt # default mode is release
	@rm -rf _build
	@mkdir _build
	@cd _build ; cmake ..

.PHONY: rebuild-coverage
rebuild-coverage:
	@rm -rf _build
	@mkdir _build
	@cd _build ; cmake -DCOVERAGE_MODE=ON ..
	@make --no-print-directory test

.PHONY: rebuild-debug
rebuild-debug:
	@rm -rf _build
	@mkdir _build
	@cd _build ; cmake -DDEBUG_MODE=ON ..
	@make --no-print-directory

.PHONY: rebuild-release
rebuild-release:
	@rm -rf _build
	@mkdir _build
	@cd _build ; cmake ..
	@make --no-print-directory

.PHONY: clean
clean:
	@rm -rf _build
	@rm -rf _build_coverage
	@rm -rf _test
	@rm -rf _aamutsel

# ==================================================================================================
#  SUBMODULES
# ==================================================================================================
.PHONY: modules
modules:
	git submodule update --init --recursive

.PHONY: modules-latest
modules-latest:
	git submodule foreach git pull origin master


# ==================================================================================================
#  CODE QUALITY
# ==================================================================================================
.PHONY: format # Requires: clang-format
format:
	clang-format -i `find src -name *.*pp`

# ==================================================================================================
#  TESTING
# ==================================================================================================
.PHONY: test
test: all
	@echo "\n\e[35m\e[1m== Tree test ==================================================================\e[0m"
	_build/tree_test
	@echo "\n\n\e[35m\e[1m== All sequential tests =======================================================\e[0m"
	_build/all_tests
	@echo "\n\n\e[35m\e[1m== MPI par test ===============================================================\e[0m"
	mpirun --oversubscribe -np 3 _build/mpi_par_test
	@echo "\n\n\e[35m\e[1m== Globom ====================================================================\e[0m"
	make --no-print-directory globom
	@echo "\n\n\e[35m\e[1m== Globom MPI ====================================================================\e[0m"
	make --no-print-directory globom

.PHONY: globom
globom: all
	@_build/globom -a data/toy_bl.ali -t data/toy_bl.nhx -u 50 tmp

.PHONY: globomMPI
globomMPI: all
	mpirun --oversubscribe -np 3 valgrind _build/globomMPI -a data/toy_bl.ali -t data/toy_bl.nhx -u 50 tmp

.PHONY: globom_dbg
globom_dbg: all
	@gdb --args _build/globom -a data/toy_bl.ali -t data/toy_bl.nhx -u 3 tmp
