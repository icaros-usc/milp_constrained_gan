#include <ilcplex/cplex.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

struct solve_callback_data
{
    // solve_callback_data():node_lp(0)
    CPXLPptr node_lp;
};

int solve_func(CPXCENVptr env,
           void *cbdata,
           int wherefrom,
           void *cbhandle,
           int *useraction_p){

    int status;
    
    struct solve_callback_data* my_data = (struct solve_callback_data*)cbhandle;

    CPXCLPptr node_lp;
    status = CPXgetcallbacknodelp(env, cbdata, wherefrom, &node_lp);

    my_data->node_lp = CPXcloneprob(env, node_lp, &status);

    // int numcols = CPXgetnumcols(env, my_data->node_lp);
    // int* feas = malloc(sizeof(int) * numcols);
    // status = CPXgetcallbacknodeintfeas(env, cbdata, wherefrom, feas, 0, numcols-1);
    // free(feas);

    // int num_infeas = 0;
    // for (int i = 0; i < numcols; ++i)
    // {
    //     num_infeas++;
    // }
    // printf("num_infeas:%d\n", num_infeas);
    // int numrows = CPXgetnumrows(env, my_data->node_lp);
    // printf("num new rows %d\n", numrows);
    // printf("num new cols %d\n", numcols);
    // if ( status ) {
    //     char errmsg[CPXMESSAGEBUFSIZE];
    //     CPXgeterrorstring(env, status, errmsg);
    //     fprintf(stderr, "%s", errmsg);
    //     return 0;
    // }
     // CPXgetcallbacknodeinfo(env, cbdata, wherefrom, int nodeindex, int whichinfo, void * result_p )
    // CPXLPptr node_lp = CPXcloneprob(env, mip, &status);
    return 0;
}



CPXLPptr gomory_solver_no_branching(CPXENVptr env, CPXLPptr mip){
    // create subproblem
    // set num nodes to be 1
    // optimize with gomory cuts
    // extract cuts
    int status;

    int sol_stat;
    int numcols;
    int cur_numrows;
    int orig_numrows;
    // double *x = (double *) malloc (numcols * sizeof(double));
    // CPXsolution(env, mip, &solstat, &objval, x, NULL, NULL, NULL);

    CPXLPptr sub_mip = CPXcloneprob(env, mip, &status);

    int is_solving = 1;

    int iters_rows_unchanged = 0;

    struct solve_callback_data solve_data;

    status = CPXsetsolvecallbackfunc(env, solve_func, &solve_data);
    while(is_solving){
        orig_numrows = CPXgetnumrows(env, sub_mip);
        // solve root node
        status = CPXmipopt(env, sub_mip);

        // status = CPXsetintparam(env, CPX_PARAM_RANDOMSEED, rand());

        // get status
        sol_stat = CPXgetstat(env, sub_mip);

        if(sol_stat == CPXMIP_NODE_LIM_INFEAS){
            // if node limit reached, restart with added cuts

            // use final lp relaxation of root node
            // add integrality constraints on variables
            numcols = CPXgetnumcols(env, sub_mip);
            char* xctype = malloc(sizeof(char) * numcols);
            status = CPXgetctype(env, sub_mip, xctype, 0, numcols - 1);
            status = CPXfreeprob(env, &sub_mip);
            status = CPXcopyctype(env, solve_data.node_lp, xctype);
            free(xctype);
            sub_mip = CPXcloneprob(env, solve_data.node_lp, &status);
            cur_numrows = CPXgetnumrows(env, sub_mip);

            if(cur_numrows == orig_numrows){
                iters_rows_unchanged ++;
            }else{
                iters_rows_unchanged = 0;
            }

            if(iters_rows_unchanged > 5){
                is_solving = 0;
            }

            status = CPXfreeprob(env, &solve_data.node_lp);
        }else{
            // otherwise terminate and return final MIP
            // TODO: return final MIP in a reasonable form, possibly just write to file
            numcols = CPXgetnumcols(env, sub_mip);
            char* xctype = malloc(sizeof(char) * numcols);
            status = CPXgetctype(env, sub_mip, xctype, 0, numcols - 1);
            status = CPXfreeprob(env, &sub_mip);
            status = CPXcopyctype(env, solve_data.node_lp, xctype);
            free(xctype);
            sub_mip = CPXcloneprob(env, solve_data.node_lp, &status);

            is_solving = 0;
            status = CPXfreeprob(env, &solve_data.node_lp);
        }
    }
    // printf("output num rows:%d\n", CPXgetnumrows(env, sub_mip));
    // use sub_mip as the overall LP relaxation we want to use
    status = CPXdelnames(env, sub_mip);
    // status = CPXwriteprob(env, sub_mip, "test_output.mps", NULL);
    return sub_mip;
}


int main(int argc, char* argv[]) {
    int status;

    if(argc < 3){
      printf("only %d arguments, need input mip and output mip\n", argc);
      return 1;
    }

    CPXENVptr env = CPXopenCPLEX(&status);
    if ( status ) {
        char errmsg[CPXMESSAGEBUFSIZE];
        CPXgeterrorstring(env, status, errmsg);
        fprintf(stderr, "%s", errmsg);
        return 0;
    }

    status = CPXsetintparam(env, CPX_PARAM_NODELIM, 1);
    // turn off presolve
    status = CPXsetintparam(env, CPX_PARAM_PREIND, 0);

    // set int tolerance to be higher
    status = CPXsetdblparam(env, CPXPARAM_MIP_Tolerances_Integrality, 0.1);
    
    // turn off heuristics
    status = CPXsetintparam(env, CPX_PARAM_HEURFREQ, -1);

    // disable all cuts except gomory
    // status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_Cliques, -1);
    // status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_Covers, -1);
    // status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_Disjunctive, -1);
    // status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_FlowCovers, -1);
    // status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_PathCut, -1);
    // status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_GUBCovers, -1);
    // status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_MIRCut, 2);
    // status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_Implied, -1);
    // status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_LocalImplied, -1);
    // status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_LiftProj, -1);
    // status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_RLT, -1);
    // status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_ZeroHalfCut, -1);

    // turn off disjunctive cuts
    status = CPXsetintparam(env, CPX_PARAM_DISJCUTS, -1);

    status = CPXsetintparam(env, CPX_PARAM_SCRIND, CPX_OFF);

    status = CPXsetdblparam(env, CPXPARAM_MIP_Limits_CutsFactor, 99999);

    status = CPXsetintparam(env, CPX_PARAM_FRACPASS, 99999);

    // enable gomory cuts
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_Gomory, 2);

    status = CPXsetintparam(env, CPXPARAM_MIP_Limits_GomoryCand, 99999);
    // status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_Gomory, -1);

    // turn off bqp cuts (consider turning this on for MIQP)
    status = CPXsetintparam(env, CPXPARAM_MIP_Cuts_BQP, -1);

    CPXLPptr mip = CPXcreateprob(env, &status, "gomory_instance");

    status = CPXreadcopyprob(env, mip, argv[1], NULL);

    if ( status ) {
        char errmsg[CPXMESSAGEBUFSIZE];
        CPXgeterrorstring(env, status, errmsg);
        fprintf(stderr, "%s", errmsg);
        return 0;
    }

    // printf("input num rows:%d\n", CPXgetnumrows(env, mip));
    // printf("sending to gomory solver\n");
    CPXLPptr output_mip = gomory_solver_no_branching(env, mip);
    printf("writing to %s\n", argv[2]);
    status = CPXwriteprob(env, output_mip, argv[2], NULL);

    return 0;
}


