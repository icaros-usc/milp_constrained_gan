/*
c_solve_gomory.c

solves a MIP via gomory cuts then passes back the following information about the optimal solution:
for now just focus on passing back solution x
- generated cutting planes
- G, h, A, b
- MIP solution (?)
- dual values (?)

This is a useful resource for working with the numpy api:
https://docs.scipy.org/doc/numpy-1.15.1/reference/c-api.array.html

*/

#define PY_ARRAY_UNIQUE_SYMBOL MYLIBRARY_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL MYLIBRARY_UFUNC_API
#include "numpy/arrayobject.h"
#include <ilcplex/cplex.h>


// technically we should have a sustained state or problem instance
// this instance should be initialized in c++ given G, h, A, b of the problem
// for now just re-write the problem each time

// declare struct to pass constraint data back to optimization algorithm
struct callback_data
{
    // tracking:
    // subject to Gz <= h
    // Az  = b
    PyArrayObject* G;
    PyArrayObject* h;
    PyArrayObject* A;
    PyArrayObject* b;
};

int solve_callback_func(CPXCENVptr env,
                    void *cbdata,
                    int wherefrom,
                    void *cbhandle,
                    int *useraction_p){
    printf("in solving callback\n");
    import_array();
    int status;
    CPXLPptr callback_lp;

    struct callback_data* my_data = (struct callback_data*)cbhandle;

    status = CPXgetcallbacknodelp(env, cbdata, wherefrom, &callback_lp);
    status = CPXwriteprob(env, callback_lp, "/Users/aaronferber/Desktop/temp_1234_cuts.mps", "mps");

    
    int numcols = CPXgetnumcols(env, callback_lp);
    int numrows = CPXgetnumrows(env, callback_lp);
    // printf("problem has %d rows and %d columns\n", numrows, numcols);

    char *con_sense = (char *)malloc(sizeof(char)*numrows);

    int row_nzcnt;
    int row_surplus;
    int rmatspace;

    int * rmatbeg = (int *)malloc(sizeof(int)*numrows);
    int * rmatind;
    double * rmatval;

    status = CPXgetrows(env, callback_lp, &row_nzcnt, rmatbeg, rmatind,
                              rmatval, 0, &row_surplus, 0,
                              numrows-1);
    
    rmatspace = -row_surplus;
    rmatind = (int *)malloc(sizeof(int)*rmatspace);
    rmatval = (double *)malloc(sizeof(double)*rmatspace);

    status = CPXgetrows(env, callback_lp, &row_nzcnt, rmatbeg, rmatind,
                              rmatval, rmatspace, &row_surplus, 0,
                              numrows-1);
    int cur_var;
    double cur_coef;
    
    int eq_cons = 0;
    int lt_cons = 0;
    CPXgetsense(env, callback_lp, con_sense, 0, numrows-1);

    double *rhs = (double*)malloc(sizeof(double)*numrows);
    status = CPXgetrhs(env, callback_lp, rhs, 0, numrows - 1);

    // resize input arrays as needed
    int num_eq = 0;     // number of equality constraints
    int num_ineq = 0;   // number of inequality constraints
    for(int cur_con = 0; cur_con < numrows; cur_con++){
        if(con_sense[cur_con] == 'E'){
            num_eq++;
        }else{
            num_ineq++;
        }
    }
    printf("resizing matrices\n");
    // resize A, b, G, h to account for potential new rows
    npy_intp* new_A_dims[2];
    new_A_dims[0] = num_eq;
    new_A_dims[1] = numcols;
    my_data->A = (PyArrayObject *) PyArray_ZEROS(2, new_A_dims,
        NPY_DOUBLE, 0);


    npy_intp* new_b_dims[1];
    new_b_dims[0] = num_eq;
    my_data->b = (PyArrayObject *) PyArray_ZEROS(1, new_b_dims, 
        NPY_DOUBLE, 0);
    
    npy_intp* new_G_dims[2];
    new_G_dims[0] = num_ineq;
    new_G_dims[1] = numcols;
    my_data->G = (PyArrayObject *) PyArray_ZEROS(2, new_G_dims, 
        NPY_DOUBLE, 0);

    npy_intp* new_h_dims[1];
    new_h_dims[0] = num_ineq;
    my_data->h = (PyArrayObject *) PyArray_ZEROS(1, new_h_dims, 
        NPY_DOUBLE, 0);
    // printf("done resizing matrices\n");

    // set constraint data
    for(int cur_con=0; cur_con < numrows; cur_con++){
        int maxk = (cur_con < numrows - 1) ? rmatbeg[cur_con + 1] : rmatspace;
        for (int k = rmatbeg[cur_con]; k < maxk; k++){
            cur_var = rmatind[k];
            cur_coef = rmatval[k];
            if(con_sense[cur_con] == 'G'){
                // if >= constraint, need to negate coefficient
                PyArray_SETITEM(my_data->G, PyArray_GETPTR2(my_data->G, lt_cons, cur_var), PyFloat_FromDouble(-cur_coef));
            }
            else if(con_sense[cur_con] == 'L'){
                // if <= constraint, just store data
                PyArray_SETITEM(my_data->G, PyArray_GETPTR2(my_data->G, lt_cons, cur_var), PyFloat_FromDouble(cur_coef));
            }
            else if(con_sense[cur_con] == 'E'){
                // if == constraint, store into A and b
                PyArray_SETITEM(my_data->A, PyArray_GETPTR2(my_data->A, eq_cons, cur_var), PyFloat_FromDouble(cur_coef));
            }
        }
        if(con_sense[cur_con] == 'G'){
            // if >= constraint, need to negate coefficient
            PyArray_SETITEM(my_data->h, PyArray_GETPTR1(my_data->h, lt_cons), PyFloat_FromDouble(-rhs[cur_con]));
            lt_cons++;
        }
        else if(con_sense[cur_con] == 'L'){
            // if <= constraint, just store data
            PyArray_SETITEM(my_data->h, PyArray_GETPTR1(my_data->h, lt_cons), PyFloat_FromDouble(rhs[cur_con]));
            lt_cons++;
        }
        else if(con_sense[cur_con] == 'E'){
            // if == constraint, store into A and b
            PyArray_SETITEM(my_data->b, PyArray_GETPTR1(my_data->b, eq_cons), PyFloat_FromDouble(rhs[cur_con]));
            eq_cons++;
        }
    }
    // printf("num eq cons:%d\n", eq_cons);
    // printf("num lt cons:%d\n", lt_cons);

    // Py_INCREF(my_data->G);
    // Py_INCREF(my_data->h);
    // Py_INCREF(my_data->A);
    // Py_INCREF(my_data->b);

    free(rmatbeg);
    free(rmatind);
    free(rmatval);
    free(con_sense);
    free(rhs);

    return 0;
}

// first let's just try passing an A matrix
// then do c, b, direction, and variable type
int c_gomory_solver(
        PyArrayObject* obj_coef_c,
        PyArrayObject* var_type,
        PyArrayObject* G,
        PyArrayObject* h,
        PyArrayObject* A,
        PyArrayObject* b,

        PyArrayObject* new_G,
        PyArrayObject* new_h,
        PyArrayObject* new_A,
        PyArrayObject* new_b
        ) {

    int status = 0;
    double obj_val = -1;


    // initialize cplex environment
    CPXENVptr env = CPXopenCPLEX(&status);


    // set params for cutting planes only
    // set max of 1 nodes
    status = CPXsetintparam(env, CPX_PARAM_NODELIM, 1);
    // turn off presolve
    status = CPXsetintparam(env, CPX_PARAM_PREIND, 0);
    // turn off heuristics
    status = CPXsetintparam(env, CPX_PARAM_HEURFREQ, -1);

    // disable all cuts except gomory
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_Cliques, -1);
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_Covers, -1);
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_Disjunctive, -1);
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_FlowCovers, -1);
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_PathCut, -1);
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_GUBCovers, -1);
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_MIRCut, -1);
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_Implied, -1);
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_LocalImplied, -1);
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_LiftProj, -1);
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_RLT, -1);
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_ZeroHalfCut, -1);

    status = CPXsetintparam(env, CPX_PARAM_SCRIND, CPX_OFF);

    // enable gomory cuts
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_Gomory, 2);
    status = CPXsetintparam(env, CPXPARAM_MIP_Limits_GomoryCand, 99999);
    // status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_Gomory, -1);
    // turn off bqp cuts (consider turning this on for MIQP)
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_BQP, -1);

    // initialize MIP
    CPXLPptr lp = CPXcreateprob(env, &status, "test_problem");

    // initialize struct to collect data from final callback
    struct callback_data cb_data;

    status = CPXsetincumbentcallbackfunc(env,
                                   solve_callback_func,
                                   (void *)&cb_data);


    // construct problem from function input
    // printf("constructing problem\n");

    // get number of rows of different types from G and A inputs
    npy_intp* G_dims = PyArray_DIMS(G);
    npy_intp* A_dims = PyArray_DIMS(A);
    int num_lt = G_dims[0];
    int num_eq = A_dims[0];

    int rcnt = num_eq + num_lt;

    // construct total rhs from h stacked on b
    double * rhs = (double*)malloc(sizeof(double)*rcnt);
    char * sense = (char*)malloc(sizeof(char)*rcnt);
    for (int i = 0; i < num_lt; ++i)
    {
        rhs[i] = PyFloat_AsDouble(PyArray_GETITEM(h, PyArray_GETPTR1(h, i)));
        sense[i] = 'L';
    }
    for (int i = 0; i < num_eq; ++i)
    {
        rhs[i+num_lt] = PyFloat_AsDouble(PyArray_GETITEM(b, PyArray_GETPTR1(b, i)));
        sense[i+num_lt] = 'E';
    }
    CPXnewrows(env, lp, rcnt, rhs, sense, NULL, NULL);

    // collect variable information from input numpy matrices
    npy_intp* obj_coef_dims = PyArray_DIMS(obj_coef_c);
    int ccnt = obj_coef_dims[0];
    double * obj = (double *)malloc(sizeof(double)*ccnt);
    double * lb = (double *)malloc(sizeof(double)*ccnt);
    char * xctype = (char *)malloc(sizeof(char)*ccnt);

    for (int i = 0; i < ccnt; ++i)
    {
        obj[i] = PyFloat_AsDouble(PyArray_GETITEM(obj_coef_c, PyArray_GETPTR1(obj_coef_c, i)));
        lb[i] = -CPX_INFBOUND;
        xctype[i] = PyBytes_AsString(PyArray_GETITEM(var_type, PyArray_GETPTR1(var_type, i)))[0];
    }

    // add variables
    CPXnewcols(env, lp, ccnt, obj, lb, NULL, xctype, NULL);

    // set individual constraint coefficients
    for (int col_ind = 0; col_ind < ccnt; ++col_ind)
    {
        for (int lt_ind = 0; lt_ind < num_lt; ++lt_ind)
        {
            CPXchgcoef(env, lp, lt_ind, col_ind,
            PyFloat_AsDouble(PyArray_GETITEM(G, PyArray_GETPTR2(G, lt_ind, col_ind))));
        }
        for (int eq_ind = 0; eq_ind < num_eq; ++eq_ind)
        {
            CPXchgcoef(env, lp, eq_ind + num_lt, col_ind,
            PyFloat_AsDouble(PyArray_GETITEM(A, PyArray_GETPTR2(A, eq_ind, col_ind))));
        }
    }


    // set problem to minimize
    CPXchgobjsen(env, lp, CPX_MIN);

    int numcols = CPXgetnumcols(env, lp);
    int numrows = CPXgetnumrows(env, lp);
    // printf("original problem has %d rows and %d columns\n", numrows, numcols);

    // solve MIP
    printf("solving problem\n");

    // need to write problem to disk and then read for some reason
    // it doesn't work if this is not there
    status = CPXwriteprob(env, lp, "/Users/aaronferber/Desktop/temp_1234.mps", "mps");
    status = CPXreadcopyprob (env, lp, "/Users/aaronferber/Desktop/temp_1234.mps", "mps");
    status = CPXmipopt(env, lp);
    
    status = CPXgetstat(env, lp);
    printf("status %d\n", status);
    // TODO: error handling when problem infeasible
    printf("solved problem\n");

    status = CPXgetobjval(env, lp, &obj_val);
    printf("obj val %d\n", obj_val);

    int node_cnt = CPXgetnodecnt(env, lp);

    // printf("setting return data\n");
    printf("G %d\n", cb_data.G);
    *new_G = *cb_data.G;
    *new_h = *cb_data.h;
    *new_A = *cb_data.A;
    *new_b = *cb_data.b;

    // need to incref so that we do not deallocate when we leave scope
    // probably we need to do this for the other data as well
    Py_INCREF(new_G);
    Py_INCREF(new_h);
    Py_INCREF(new_A);
    Py_INCREF(new_b);

    // printf("incref data\n");

    // printf("freeing data\n");

    // free constraint data
    free(rhs);
    free(sense);

    // free variable data
    free(obj);
    free(lb);
    free(xctype);
    // printf("freed data\n");
    Py_RETURN_NONE;
}


