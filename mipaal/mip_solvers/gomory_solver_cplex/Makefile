## Stefano Gualandi, stefano.gualandi@gmail.com
## February, 2013

OPTFLAG         = -DIL_STD -Wall -fPIC -fexceptions -DNDEBUG 
LDFLAGS 	       = -Wl,--no-as-needed -ldl -DNDEBUG -lm -pthread -ldl

COMPILER        = gcc ${OPTFLAG}
LINKER          = gcc ${LDFLAGS}

# CPLEX directory  (SET YOUR OWN CPLEX DIRECTORIES)
CPLEX_HOME     = /home/hejia/Tools/cplex/CPLEX_Studio1210/cplex/
CPLEX_INC      = ${CPLEX_HOME}/include/
CPLEX_LIB      = ${CPLEX_HOME}/lib/x86-64_linux/static_pic/ -lcplex

LINKED_DIRECTORIES =-L${CPLEX_LIB}

# Compile the main file
cpx_gomory: cpx_gomory.c
	${COMPILER} -c cpx_gomory.c -o cpx_gomory.o -I${CPLEX_INC}
	${LINKER} -o cpx_gomory cpx_gomory.o ${LINKED_DIRECTORIES}

gomory_limited: gomory_limited.c
	${COMPILER} -c gomory_limited.c -o gomory_limited.o -I${CPLEX_INC}
	${LINKER} -o gomory_limited gomory_limited.o ${LINKED_DIRECTORIES}


cg_solver: cg_solver.c
	${COMPILER} -c cg_solver.c -o cg_solver.o -I${CPLEX_INC}
	${LINKER} -o cg_solver cg_solver.o ${LINKED_DIRECTORIES}

gomory: gomory.c
	${COMPILER} -c gomory.c -o gomory.o -I${CPLEX_INC}
	${LINKER} -o gomory gomory.o ${LINKED_DIRECTORIES} 

clean::
	rm -f *.o
	rm -f ${LIB}/*.o
	rm -f *~
	rm -f ${SRC}/*~ ${INCLUDE}/*~
	rm cpx_gomory

