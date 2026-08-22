#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <nvector/nvector_serial.h>

/* Matrices */
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_band.h>
#include <sunmatrix/sunmatrix_sparse.h>

/* Direct linear solvers */
#include <sunlinsol/sunlinsol_dense.h>
#include <sunlinsol/sunlinsol_band.h>
#ifdef SUNDIALS_KLU_ENABLED
#include <sunlinsol/sunlinsol_klu.h>
#endif

/* Iterative linear solvers */
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunlinsol/sunlinsol_spbcgs.h>
#include <sunlinsol/sunlinsol_sptfqmr.h>

/* ODE / DAE solvers (sensitivity-capable variants) */
#include <cvode/cvode.h>
#include <cvodes/cvodes.h>
#include <cvodes/cvodes_ls.h>
#include <ida/ida.h>
#include <idas/idas.h>
#include <idas/idas_ls.h>
#include <kinsol/kinsol.h>

/* ARKode */
#include <arkode/arkode.h>
#include <arkode/arkode_arkstep.h>
#include <arkode/arkode_ls.h>
