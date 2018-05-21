

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../admm.h"

#include <math.h>





/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


/**
* @brief The proximal update for column max-normalization. This routine
*        normalizes each factor column *if* the norm grows larger than 1.
*
* @param[out] primal The row-major matrix to update.
* @param nrows The number of rows in primal.
* @param ncols The number of columns in primal.
* @param offset Not used.
* @param data Not used.
* @param rho Not used.
* @param should_parallelize Should be true.
*/
void splatt_maxcolnorm_prox_old(
    val_t * primal,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const offset,
    void * data,
    val_t const rho,
    bool const should_parallelize)
{
  assert(should_parallelize);

  #pragma omp parallel for schedule(dynamic, 1)
  for(idx_t j=0; j < ncols; ++j) {
    double norm = 0.;

    for(idx_t i=0; i < nrows; ++i) {
      idx_t const index = j + (i*ncols);
      norm += primal[index] * primal[index];
    }

    norm = sqrt(norm);
    if(norm <= 1.) {
      continue;
    }

    for(idx_t i=0; i < nrows; ++i) {
      idx_t const index = j + (i*ncols);
      primal[index] /= norm;
    }
  }
}


void splatt_maxcolnorm_prox(
    val_t * primal,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const offset,
    void * data,
    val_t const rho,
    bool const should_parallelize)
{
  assert(should_parallelize);

  val_t * restrict A = primal;

  // column-wise normalization of row-major matrix

  #pragma omp parallel
  {
  double * restrict norms = (double*) splatt_malloc(nrows*sizeof(double));
  memset(norms, 0, nrows*sizeof(double));

  #pragma omp for schedule(static)
  for (idx_t i=0; i < nrows; ++i) {
    #pragma simd
    for (idx_t j=0; j < ncols; ++j) {
      idx_t const idx = i*ncols + j;
      norms[j] += A[idx] * A[idx];
    }
  }

  #pragma omp barrier
  thread_allreduce(norms, nrows, SPLATT_REDUCE_SUM);


  // row sums are copies on every thread, thus every thread
  // finishes computing the norm
  for (idx_t j=0; j < ncols; ++j) {
    norms[j] = sqrt(norms[j]);
    //all &= (norms[j] <= 1.);
    norms[j] = (norms[j] > 1.) ? norms[j] : 1.;
  }

#pragma omp for schedule(static)
  for (idx_t i=0; i < nrows; ++i) {
    #pragma simd
    for (idx_t j=0; j < ncols; ++j) {
      idx_t const idx = i*ncols + j;
      A[idx] /= norms[j];
    }
  }

  splatt_free(norms);

  }

}


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/


splatt_error_type splatt_register_maxcolnorm(
    splatt_cpd_opts * opts,
    splatt_idx_t const * const modes_included,
    splatt_idx_t const num_modes)
{
  for(idx_t m = 0; m < num_modes; ++m) {
    idx_t const mode = modes_included[m];

    splatt_cpd_constraint * con = splatt_alloc_constraint(SPLATT_CON_ADMM);

    /* only fill the details that are used */
    con->prox_func = splatt_maxcolnorm_prox;

    /* set hints to assist optimizations */
    con->hints.row_separable     = false;
    con->hints.sparsity_inducing = false;

    sprintf(con->description, "MAX-COL-NORM");

    /* add to the CPD factorization */
    splatt_register_constraint(opts, mode, con);
  }

  return SPLATT_SUCCESS;
}


