/* specfunc/test_alf.c
 * 
 * Copyright (C) 2023, 2024, 2025, 2026 Patrick Alken
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#include <config.h>
#include <stdarg.h>
#include <gsl/gsl_test.h>
#include <gsl/gsl_sf.h>
#include <assert.h>
#include "test_sf.h"

#include "test_alf_e.c"

static double
test_alf_dx(const size_t l)
{
  const double dx_max = 0.4;
  double dx;

  if (l < 1000)
    dx = exp((double)l / 1000.0) / exp(2.0);
  else
    dx = dx_max;

  return dx;
}

/*
test_alf_sum()
  This routine computes the sum:

  Sum_{m=0}^l [P(l,m)(x)]^2

This sum should equate to 1.0 for Schmidt semi-normalized
ALFs for all l.
*/

static double
test_alf_sum(const size_t lmax, const size_t l, double *p)
{
  double sum = 0.0;
  size_t m;

  for (m = 0; m <= l; ++m)
    {
      size_t idx = gsl_sf_alf_array_index(l, m, lmax);
      sum += p[idx] * p[idx];
    }

  return sum;
}

/*
test_alf_sum_deriv()
  This routine computes the sum:

  Sum_{m=0}^l P(l,m)(x) * dP(l,m)/dx

which should equal 0 in the case of Schmidt normalized ALFs.
*/

static double
test_alf_sum_deriv(const size_t lmax, const size_t l, double *p, double *dp)
{
  double sum = 0.0;
  size_t m;

  for (m = 0; m <= l; ++m)
    {
      size_t idx = gsl_sf_alf_array_index(l, m, lmax);
      sum += p[idx] * dp[idx];
    }

  return sum;
}

/*
test_alf_sum_deriv2()
  This routine computes the sum:

  Sum_{m=0}^l P(l,m)(x) * dP(l,m)/dx

which should equal 0 in the case of Schmidt normalized ALFs.
*/

static double
test_alf_sum_deriv2(const size_t lmax, const size_t l, double *p, double *dp, double *d2p)
{
  double sum = 0.0;
  size_t m;

  for (m = 0; m <= l; ++m)
    {
      size_t idx = gsl_sf_alf_array_index(l, m, lmax);
      sum += dp[idx] * dp[idx] + p[idx] * d2p[idx];
    }

  return sum;
}

static void
test_value(const size_t lmax, const size_t mmax, const size_t l, const size_t m,
           const double *p, const double expected, const double tol,
           const char *desc, ...)
{
  size_t idx = gsl_sf_alf_array_index(l, m, lmax);
  va_list args;
  char buf[512];

  if (l > lmax || m > mmax)
    return;

  va_start(args, desc);

  vsnprintf(buf, sizeof(buf), desc, args);

  gsl_test_rel(expected, p[idx], tol, "%s lmax=%zu mmax=%zu l=%zu m=%zu", buf, lmax, mmax, l, m);

  va_end(args);
}

/* Y_{lm} = factor * S_{lm} */
static double
test_factor_spharm(const size_t l, const size_t m)
{
  double factor = sqrt( (2.0 * l + 1.0) / 4.0 / M_PI);

  if (m == 0)
    return factor;
  else
    return (factor / sqrt(2.0));
}

/* N_{lm} = factor * S_{lm} */
static double
test_factor_full(const size_t l, const size_t m)
{
  double factor = sqrt(l + 0.5);

  if (m == 0)
    return factor;
  else
    return (factor / sqrt(2.0));
}

/* R_{lm} = factor * S_{lm} */
static double
test_factor_fourpi(const size_t l, const size_t m)
{
  double factor = sqrt(2.0 * l + 1.0);

  (void) m;

  return factor;
}

/* test that p = factor * p_expected */
static int
test_legendre_compare(const double tol, const size_t lmax, const size_t mmax,
                      const double *p_expected, const double *p, const double x,
                      double (*factor)(const size_t l, const size_t m),
                      const char *desc, const char *desc2)
{
  size_t l, m;

  for (m = 0; m <= mmax; ++m)
    {
      for (l = m; l <= lmax; ++l)
        {
          size_t idx = gsl_sf_alf_array_index(l, m, lmax);
          double fac = (*factor)(l, m);

          if (fabs(p_expected[idx]) < GSL_DBL_MIN)
            continue;

          gsl_test_rel(p[idx] / fac, p_expected[idx], tol,
                       "%s %s x=%g l=%zu m=%zu", desc, desc2, x, l, m);
        }
    }

  return 0;
}

/* test gsl_sf_alf_array function */
static int
test_alf_schmidt1(const double tol, const size_t lmax, const size_t mmax, const double x,
                  const size_t expected_L, const double expected_S[], const char * desc)
{
  const size_t plm_size = gsl_sf_alf_array_size(lmax, mmax);
  double * Plm = malloc(plm_size * sizeof(double));
  size_t l, m;

  /* precompute ALF factors */
  gsl_sf_alf_precompute(GSL_SF_ALF_SCHMIDT, lmax, mmax, 0, Plm);

  /* compute ALFs */
  gsl_sf_alf_array(lmax, mmax, x, Plm);

  for (l = 0; l <= GSL_MIN(lmax, expected_L); ++l)
    {
      for (m = 0; m <= GSL_MIN(l, mmax); ++m)
        {
          size_t idx = gsl_sf_alf_array_index(l, m, expected_L);
          test_value(lmax, mmax, l, m, Plm, expected_S[idx], tol, "[schmidt1] 0th %s x=%g", desc, x);
        }
    }

  free(Plm);

  return GSL_SUCCESS;
}

/* test gsl_sf_alf_deriv_array function */
static int
test_alf_schmidt2(const double tol, const size_t lmax, const size_t mmax, const double x,
                  const size_t expected_L, const double expected_S[],
                  const double expected_DS[], const char * desc)
{
  const size_t plm_size = gsl_sf_alf_array_size(lmax, mmax);
  const size_t dplm_size = gsl_sf_alf_nlm(lmax, mmax);
  double * Plm = malloc(plm_size * sizeof(double));
  double * dPlm = malloc(dplm_size * sizeof(double));
  size_t l, m;

  /* precompute ALF factors */
  gsl_sf_alf_precompute(GSL_SF_ALF_SCHMIDT, lmax, mmax, 0, Plm);

  /* compute ALFs */
  gsl_sf_alf_deriv_array(lmax, mmax, x, Plm, dPlm);

  for (l = 0; l <= GSL_MIN(lmax, expected_L); ++l)
    {
      for (m = 0; m <= GSL_MIN(l, mmax); ++m)
        {
          size_t idx = gsl_sf_alf_array_index(l, m, expected_L);
          test_value(lmax, mmax, l, m, Plm, expected_S[idx], tol, "[schmidt2] 0th %s x=%g", desc, x);
          test_value(lmax, mmax, l, m, dPlm, expected_DS[idx], tol, "[schmidt2] 1st %s x=%g", desc, x);
        }
    }

  free(Plm);
  free(dPlm);

  return GSL_SUCCESS;
}

/* test gsl_sf_alf_deriv2_array function */
static int
test_alf_schmidt3(const double tol, const size_t lmax, const size_t mmax, const double x,
                  const size_t expected_L, const double expected_S[],
                  const double expected_DS[], const double expected_DDS[],
                  const char * desc)
{
  const size_t plm_size = gsl_sf_alf_array_size(lmax, mmax);
  const size_t dplm_size = gsl_sf_alf_nlm(lmax, mmax);
  double * Plm = malloc(plm_size * sizeof(double));
  double * dPlm = malloc(dplm_size * sizeof(double));
  double * d2Plm = malloc(dplm_size * sizeof(double));
  size_t l, m;

  /* precompute ALF factors */
  gsl_sf_alf_precompute(GSL_SF_ALF_SCHMIDT, lmax, mmax, 0, Plm);

  /* compute ALFs */
  gsl_sf_alf_deriv2_array(lmax, mmax, x, Plm, dPlm, d2Plm);

  for (l = 0; l <= GSL_MIN(lmax, expected_L); ++l)
    {
      for (m = 0; m <= GSL_MIN(l, mmax); ++m)
        {
          size_t idx = gsl_sf_alf_array_index(l, m, expected_L);
          test_value(lmax, mmax, l, m, Plm, expected_S[idx], tol, "[schmidt3] 0th %s x=%g", desc, x);
          test_value(lmax, mmax, l, m, dPlm, expected_DS[idx], tol, "[schmidt3] 1st %s x=%g", desc, x);
          test_value(lmax, mmax, l, m, d2Plm, expected_DDS[idx], tol, "[schmidt3] 2nd %s x=%g", desc, x);
        }
    }

  free(Plm);
  free(dPlm);
  free(d2Plm);

  return GSL_SUCCESS;
}

static int
test_alf_schmidt(const size_t lmax, const size_t mmax, const size_t flags, const char *desc)
{
  int s = 0;
  const double tol = 1.0e-10;
  const size_t nlm = gsl_sf_alf_nlm(lmax, mmax);
  const size_t plm_size = gsl_sf_alf_array_size(lmax, mmax);
  const size_t dplm_size = nlm;
  double * Plm = malloc(plm_size * sizeof(double));
  double * Plm2 = malloc(plm_size * sizeof(double));
  double * dPlm = malloc(dplm_size * sizeof(double));
  double * dPlm2 = malloc(dplm_size * sizeof(double));
  double * d2Plm = malloc(dplm_size * sizeof(double));
  double * Plm_theta = malloc(plm_size * sizeof(double));
  double * dPlm_theta = malloc(dplm_size * sizeof(double));
  double * d2Plm_theta = malloc(dplm_size * sizeof(double));
  double x, dx;

  /* test specific values - the alf.nb Mathematica notebook is used to generated the expected results */

  /* test gsl_sf_alf_array */
  test_alf_schmidt1(tol, lmax, mmax, -1.0,  expected_L_MNB, expected_S_m1,  desc);
  test_alf_schmidt1(tol, lmax, mmax, -0.9,  expected_L_MNB, expected_S_m09, desc);
  test_alf_schmidt1(tol, lmax, mmax, -0.2,  expected_L_MNB, expected_S_m02, desc);
  test_alf_schmidt1(tol, lmax, mmax,  0.15, expected_L_MNB, expected_S_015, desc);
  test_alf_schmidt1(tol, lmax, mmax,  0.23, expected_L_MNB, expected_S_023, desc);
  test_alf_schmidt1(tol, lmax, mmax,  0.6,  expected_L_MNB, expected_S_06,  desc);
  test_alf_schmidt1(tol, lmax, mmax,  0.75, expected_L_MNB, expected_S_075, desc);
  test_alf_schmidt1(tol, lmax, mmax,  1.0,  expected_L_MNB, expected_S_1,   desc);

  /* test gsl_sf_alf_deriv_array */
  test_alf_schmidt2(tol, lmax, mmax, -0.9,  expected_L_MNB, expected_S_m09, expected_DS_m09, desc);
  test_alf_schmidt2(tol, lmax, mmax, -0.2,  expected_L_MNB, expected_S_m02, expected_DS_m02, desc);
  test_alf_schmidt2(tol, lmax, mmax,  0.15, expected_L_MNB, expected_S_015, expected_DS_015, desc);
  test_alf_schmidt2(tol, lmax, mmax,  0.23, expected_L_MNB, expected_S_023, expected_DS_023, desc);
  test_alf_schmidt2(tol, lmax, mmax,  0.6,  expected_L_MNB, expected_S_06,  expected_DS_06,  desc);
  test_alf_schmidt2(tol, lmax, mmax,  0.75, expected_L_MNB, expected_S_075, expected_DS_075, desc);

  /* test gsl_sf_alf_deriv2_array */
  test_alf_schmidt3(tol, lmax, mmax, -0.9,  expected_L_MNB, expected_S_m09, expected_DS_m09, expected_DDS_m09, desc);
  test_alf_schmidt3(tol, lmax, mmax, -0.2,  expected_L_MNB, expected_S_m02, expected_DS_m02, expected_DDS_m02, desc);
  test_alf_schmidt3(tol, lmax, mmax,  0.15, expected_L_MNB, expected_S_015, expected_DS_015, expected_DDS_015, desc);
  test_alf_schmidt3(tol, lmax, mmax,  0.23, expected_L_MNB, expected_S_023, expected_DS_023, expected_DDS_023, desc);
  test_alf_schmidt3(tol, lmax, mmax,  0.6,  expected_L_MNB, expected_S_06,  expected_DS_06,  expected_DDS_06,  desc);
  test_alf_schmidt3(tol, lmax, mmax,  0.75, expected_L_MNB, expected_S_075, expected_DS_075, expected_DDS_075, desc);

  gsl_sf_alf_precompute(GSL_SF_ALF_SCHMIDT, lmax, mmax, 0, Plm);

  x = 0.1;
  gsl_sf_alf_array(lmax, mmax, x, Plm);
  test_value(lmax, mmax, 2700, 500, Plm, -7.421910573369699e-03, tol, "%s x=%g", desc, x);
  test_value(lmax, mmax, 2700, 2500, Plm, 2.717612388452281e-02, tol, "%s x=%g", desc, x);
  test_value(lmax, mmax, 2700, 2700, Plm, 1.887509917445211e-07, tol, "%s x=%g", desc, x);

  x = 0.35;
  gsl_sf_alf_vsh_array(lmax, mmax, x, Plm, dPlm_theta);
  test_value(lmax, mmax, 0, 0, Plm,   1.000000000000000, tol, "%s theta x=%g", desc, x);
  test_value(lmax, mmax, 1, 0, Plm,   0.350000000000000, tol, "%s theta x=%g", desc, x);
  test_value(lmax, mmax, 1, 1, Plm,   1.000000000000000, tol, "%s theta x=%g", desc, x);
  test_value(lmax, mmax, 2, 0, Plm,  -0.316250000000000, tol, "%s theta x=%g", desc, x);
  test_value(lmax, mmax, 2, 1, Plm,   0.606217782649107, tol, "%s theta x=%g", desc, x);
  test_value(lmax, mmax, 2, 2, Plm,   0.811249036979398, tol, "%s theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 0, Plm,  -0.417812500000000, tol, "%s theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 1, Plm,  -0.237294318832120, tol, "%s theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 2, Plm,   0.634902797678511, tol, "%s theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 3, Plm,   0.693724661699438, tol, "%s theta x=%g", desc, x);

  test_value(lmax, mmax, 0, 0, dPlm_theta,  0.000000000000000, tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 1, 0, dPlm_theta, -0.936749699759760, tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 1, 1, dPlm_theta,  0.350000000000000, tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 2, 0, dPlm_theta, -0.983587184747748, tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 2, 1, dPlm_theta, -1.30769835971450,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 2, 2, dPlm_theta,  0.567874325885578, tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 0, dPlm_theta,  0.544485762985360, tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 1, dPlm_theta, -1.963801854721950, tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 2, dPlm_theta, -1.14736005580474,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 3, dPlm_theta,  0.728410894784410, tol, "%s deriv theta x=%g", desc, x);

  if (lmax == mmax)
    {
      gsl_sf_alf_theta_derivk_array(lmax, Plm, dPlm_theta, d2Plm_theta);
      test_value(lmax, mmax, 0, 0, d2Plm_theta,  0.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 1, 0, d2Plm_theta, -0.350000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 1, 1, d2Plm_theta, -0.936749699759760, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 2, 0, d2Plm_theta,  2.265000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 2, 1, d2Plm_theta, -2.271497303542314, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 2, 2, d2Plm_theta, -1.307698359714502, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 3, 0, d2Plm_theta,  4.810312500000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 3, 1, d2Plm_theta,  3.147847827844545, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 3, 2, d2Plm_theta, -3.997160874745192, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 3, 3, d2Plm_theta, -1.405223343986419, tol, "%s derivk k=2 theta x=%g", desc, x);
    }

  x = 1.0;
  gsl_sf_alf_vsh_array(lmax, mmax, x, Plm, dPlm_theta);
  test_value(lmax, mmax, 0, 0, Plm,  1.000000000000000,  tol, "%s theta x=%g", desc, x);
  test_value(lmax, mmax, 1, 0, Plm,  1.000000000000000,  tol, "%s theta x=%g", desc, x);
  test_value(lmax, mmax, 2, 0, Plm,  1.000000000000000,  tol, "%s theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 0, Plm,  1.000000000000000,  tol, "%s theta x=%g", desc, x);

  test_value(lmax, mmax, 0, 0, dPlm_theta, 0.000000000000000,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 1, 0, dPlm_theta, 0.000000000000000,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 1, 1, dPlm_theta, 1.000000000000000,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 2, 0, dPlm_theta, 0.000000000000000,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 2, 1, dPlm_theta, M_SQRT3,            tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 2, 2, dPlm_theta, 0.000000000000000,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 0, dPlm_theta, 0.000000000000000,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 1, dPlm_theta, 2.44948974278318,   tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 2, dPlm_theta, 0.000000000000000,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 3, dPlm_theta, 0.000000000000000,  tol, "%s deriv theta x=%g", desc, x);

  if (lmax == mmax)
    {
      gsl_sf_alf_theta_derivk_array(lmax, Plm, dPlm_theta, d2Plm_theta);
      test_value(lmax, mmax, 0, 0, d2Plm_theta,  0.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 1, 0, d2Plm_theta, -1.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 1, 1, d2Plm_theta,  0.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 2, 0, d2Plm_theta, -3.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 2, 1, d2Plm_theta,  0.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 2, 2, d2Plm_theta,  M_SQRT3,           tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 3, 0, d2Plm_theta, -6.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 3, 1, d2Plm_theta,  0.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 3, 2, d2Plm_theta,  3.872983346207417, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 3, 3, d2Plm_theta,  0.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
    }

  x = -1.0;
  gsl_sf_alf_vsh_array(lmax, mmax, x, Plm, dPlm_theta);
  test_value(lmax, mmax, 0, 0, Plm,   1.000000000000000,  tol, "%s theta x=%g", desc, x);
  test_value(lmax, mmax, 1, 0, Plm,  -1.000000000000000,  tol, "%s theta x=%g", desc, x);
  test_value(lmax, mmax, 2, 0, Plm,   1.000000000000000,  tol, "%s theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 0, Plm,  -1.000000000000000,  tol, "%s theta x=%g", desc, x);

  test_value(lmax, mmax, 0, 0, dPlm_theta,  0.000000000000000,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 1, 0, dPlm_theta,  0.000000000000000,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 1, 1, dPlm_theta, -1.000000000000000,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 2, 0, dPlm_theta,  0.000000000000000,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 2, 1, dPlm_theta,  M_SQRT3,            tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 2, 2, dPlm_theta,  0.000000000000000,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 0, dPlm_theta,  0.000000000000000,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 1, dPlm_theta, -2.44948974278318,   tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 2, dPlm_theta,  0.000000000000000,  tol, "%s deriv theta x=%g", desc, x);
  test_value(lmax, mmax, 3, 3, dPlm_theta,  0.000000000000000,  tol, "%s deriv theta x=%g", desc, x);

  if (lmax == mmax)
    {
      gsl_sf_alf_theta_derivk_array(lmax, Plm, dPlm_theta, d2Plm_theta);
      test_value(lmax, mmax, 0, 0, d2Plm_theta,  0.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 1, 0, d2Plm_theta,  1.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 1, 1, d2Plm_theta,  0.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 2, 0, d2Plm_theta, -3.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 2, 1, d2Plm_theta,  0.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 2, 2, d2Plm_theta,  M_SQRT3,           tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 3, 0, d2Plm_theta,  6.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 3, 1, d2Plm_theta,  0.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 3, 2, d2Plm_theta, -3.872983346207417, tol, "%s derivk k=2 theta x=%g", desc, x);
      test_value(lmax, mmax, 3, 3, d2Plm_theta,  0.000000000000000, tol, "%s derivk k=2 theta x=%g", desc, x);
    }

  /* test array routines */
  dx = test_alf_dx(lmax);
  gsl_sf_alf_precompute(GSL_SF_ALF_SCHMIDT, lmax, mmax, flags, Plm);
  gsl_sf_alf_precompute(GSL_SF_ALF_SCHMIDT, lmax, mmax, flags, Plm2);
  gsl_sf_alf_precompute(GSL_SF_ALF_SCHMIDT, lmax, mmax, flags, Plm_theta);

  /* test deriv array routines */
  for (x = -1.0; x <= 1.0; x += dx)
    {
      double u = sqrt((1.0 - x) * (1.0 + x));
      size_t idx = 0;
      size_t l, m;

      s += gsl_sf_alf_array(lmax, mmax, x, Plm2);
      s += gsl_sf_alf_vsh_array(lmax, mmax, x, Plm_theta, dPlm_theta);

      if (x < 1.0 && x > -1.0)
        s += gsl_sf_alf_deriv_array(lmax, mmax, x, Plm, dPlm);

      for (m = 0; m <= mmax; ++m)
        {
          for (l = m; l <= lmax; ++l)
            {
              if (fabs(Plm2[idx]) < GSL_DBL_MIN)
                {
                  ++idx;
                  continue;
                }

              assert(idx == gsl_sf_alf_array_index(l, m, lmax));

              if (x < 1.0 && x > -1.0)
                {
                  /* check Plm = Plm2 */
                  gsl_test_rel(Plm[idx], Plm2[idx], tol, "%s deriv P x=%g lmax=%zu mmax=%zu l=%zu m=%zu",
                               desc, x, lmax, mmax, l, m);

                  /* check dPlm = -1/u * dPlm_theta */
                  gsl_test_rel(-u * dPlm[idx], dPlm_theta[idx], tol,
                               "%s deriv_theta x=%g lmax=%zu mmax=%zu l=%zu m=%zu",
                               desc, x, lmax, mmax, l, m);
                }

              if (m == 0)
                {
                  /* check Plm_theta = Plm2 */
                  gsl_test_rel(Plm_theta[idx], Plm2[idx], tol, "%s deriv theta P x=%g lmax=%zu mmax=%zu l=%zu m=%zu",
                               desc, x, lmax, mmax, l, m);
                }
              else
                {
                  /* check Plm_theta*sint = Plm2 */
                  gsl_test_rel(Plm_theta[idx]*u, Plm2[idx], tol, "%s deriv theta P x=%g lmax=%zu mmax=%zu l=%zu m=%zu",
                               desc, x, lmax, mmax, l, m);
                }

              ++idx;
            }
        }

      if (lmax == mmax && x < 1.0 && x > -1.0)
        {
          for (l = 0; l <= lmax; ++l)
            {
              double sum = test_alf_sum_deriv(lmax, l, Plm, dPlm);

              gsl_test_abs(sum, 0.0, tol,
                           "%s deriv lmax=%zu mmax=%zu l=%zu, x=%f, sum=%.12e", desc, lmax, mmax, l, x, sum);
            }
        }
    }

  /* test deriv2 array routines against deriv */
  for (x = -1.0 + dx; x < 1.0 - dx; x += dx)
    {
      size_t i;

      s += gsl_sf_alf_deriv_array(lmax, mmax, x, Plm, dPlm);
      s += gsl_sf_alf_deriv2_array(lmax, mmax, x, Plm2, dPlm2, d2Plm);

      /* check Plm == Plm2 and dPlm == dPlm2 */
      for (i = 0; i < nlm; ++i)
        {
          if (fabs(Plm[i]) < 1.0e3 * GSL_DBL_EPSILON)
            gsl_test_abs(Plm2[i], Plm[i], tol, "%s deriv2 0th i=%zu", desc, i);
          else
            gsl_test_rel(Plm2[i], Plm[i], tol, "%s deriv2 0th i=%zu", desc, i);

          if (fabs(dPlm[i]) < 1.0e3 * GSL_DBL_EPSILON)
            gsl_test_abs(dPlm2[i], dPlm[i], tol, "%s deriv2 1st i=%zu", desc, i);
          else
            gsl_test_rel(dPlm2[i], dPlm[i], tol, "%s deriv2 1st i=%zu", desc, i);
        }

      if (lmax == mmax)
        {
          size_t l;

          for (l = 0; l <= lmax; ++l)
            {
              double sum = test_alf_sum_deriv(lmax, l, Plm, dPlm);
              double sum2 = test_alf_sum_deriv(lmax, l, Plm2, dPlm2);
              double sum3 = test_alf_sum_deriv2(lmax, l, Plm2, dPlm2, d2Plm);

              gsl_test_abs(sum, 0.0, tol,
                           "%s deriv2 lmax=%zu mmax=%zu l=%zu, x=%f, sum=%.12e", desc, lmax, mmax, l, x, sum);
              gsl_test_abs(sum2, 0.0, tol,
                           "%s deriv2 lmax=%zu mmax=%zu l=%zu, x=%f, sum2=%.12e", desc, lmax, mmax, l, x, sum2);
              gsl_test_abs(sum3, 0.0, tol,
                           "%s deriv2 lmax=%zu mmax=%zu l=%zu, x=%f, sum3=%.12e", desc, lmax, mmax, l, x, sum3);
            }
        }
    }

  free(Plm);
  free(Plm2);
  free(dPlm);
  free(dPlm2);
  free(d2Plm);
  free(Plm_theta);
  free(dPlm_theta);
  free(d2Plm_theta);

  return s;
}

/* test other normalizations (other than schmidt) */
static int
test_alf_norm(const gsl_sf_alf_t norm, const size_t lmax,
              const size_t mmax, const size_t flags,
              const char *desc)
{
  int s = 0;
  const double tol = 1.0e-10;
  const size_t plm_size = gsl_sf_alf_array_size(lmax, mmax);
  const size_t dplm_size = gsl_sf_alf_nlm(lmax, mmax);
  double x, dx;
  double (*factor)(const size_t l, const size_t m) = NULL;

  double *p = malloc(sizeof(double) * plm_size);
  double *dp = malloc(sizeof(double) * dplm_size);
  double *p_schmidt = malloc(sizeof(double) * plm_size);
  double *dp_schmidt = malloc(sizeof(double) * dplm_size);

  if (norm == GSL_SF_ALF_SPHARM)
    {
      factor = &test_factor_spharm;
    }
  else if (norm == GSL_SF_ALF_FULL)
    {
      factor = &test_factor_full;

      /* test specific values (computed from GNU octave) */
      gsl_sf_alf_precompute(norm, lmax, mmax, 0, p);
      x = 0.45;
      s += gsl_sf_alf_array(lmax, mmax, x, p);
      test_value(lmax, mmax, 0, 0, p,  0.707106781186548,   tol, desc, "x=0.45");
      test_value(lmax, mmax, 1, 0, p,  0.551135192126215,   tol, desc, "x=0.45");
      test_value(lmax, mmax, 1, 1, p,  0.773385414912901,   tol, desc, "x=0.45");
      test_value(lmax, mmax, 2, 0, p, -0.310298495404022,   tol, desc, "x=0.45");
      test_value(lmax, mmax, 2, 1, p,  0.778204062248457,   tol, desc, "x=0.45");
      test_value(lmax, mmax, 2, 2, p,  0.772176054650104,   tol, desc, "x=0.45");
      test_value(lmax, mmax, 3, 0, p, -0.83661120632398589, tol, desc, "x=0.45");
      test_value(lmax, mmax, 3, 1, p,  0.00904294765791280, tol, desc, "x=0.45");
      test_value(lmax, mmax, 3, 2, p,  0.91934361403343767, tol, desc, "x=0.45");
      test_value(lmax, mmax, 3, 3, p,  0.74482641545541073, tol, desc, "x=0.45");
    }
  else if (norm == GSL_SF_ALF_FOURPI)
    {
      factor = &test_factor_fourpi;
    }

  /*
   * test the scale factors between the Schmidts and these
   * normalized functions
   */
  gsl_sf_alf_precompute(GSL_SF_ALF_SCHMIDT, lmax, mmax, flags, p_schmidt);
  gsl_sf_alf_precompute(norm, lmax, mmax, flags, p);

  dx = test_alf_dx(lmax);
  for (x = -1.0; x <= 1.0; x += dx)
    {
      s += gsl_sf_alf_array(lmax, mmax, x, p_schmidt);
      s += gsl_sf_alf_array(lmax, mmax, x, p);
      test_legendre_compare(tol, lmax, mmax, p_schmidt, p, x, factor, desc, "p");
    }

  /* test derivatives */
  for (x = -1.0 + dx; x < 1.0 - dx; x += dx)
    {
      s += gsl_sf_alf_deriv_array(lmax, mmax, x, p_schmidt, dp_schmidt);
      s += gsl_sf_alf_deriv_array(lmax, mmax, x, p, dp);
      test_legendre_compare(tol, lmax, mmax, p_schmidt, p, x, factor, desc, "deriv p");
      test_legendre_compare(tol, lmax, mmax, dp_schmidt, dp, x, factor, desc, "deriv dp");
    }

  free(p);
  free(dp);
  free(p_schmidt);
  free(dp_schmidt);

  return s;
}

/*
test_alf_unnorm()
  This routine tests the unnormalized ALFs using the relation

S(l,m)(x) = a(l,m) * P(l,m)(x)

where

a(l,0) = 1
a(l,1) = -sqrt(2)/sqrt(l * (l+1))
a(l,m+1) = a(l,m) / sqrt((l+m+1) * (l-m)), m > 1

and

S(l,m) are the Schmidt semi-normalized ALFs
*/

static int
test_alf_unnorm(const size_t lmax_orig, const size_t mmax_orig,
                const size_t flags, const char *desc)
{
  int s = 0;
  const double tol = 1.0e-10;
  const size_t lmax = GSL_MIN(lmax_orig, 140);
  const size_t mmax = GSL_MIN(lmax, mmax_orig);
  const size_t plm_size = gsl_sf_alf_array_size(lmax, mmax);
  const size_t dplm_size = gsl_sf_alf_nlm(lmax, mmax);
  size_t l, m;
  double x, dx;

  double *p = malloc(sizeof(double) * plm_size);
  double *dp = malloc(sizeof(double) * dplm_size);
  double *p2 = malloc(sizeof(double) * plm_size);
  double *p_schmidt = malloc(sizeof(double) * plm_size);
  double *dp_schmidt = malloc(sizeof(double) * dplm_size);

  gsl_sf_alf_precompute(GSL_SF_ALF_SCHMIDT, lmax, mmax, flags, p_schmidt);
  gsl_sf_alf_precompute(GSL_SF_ALF_NONE, lmax, mmax, flags, p);
  gsl_sf_alf_precompute(GSL_SF_ALF_NONE, lmax, mmax, flags, p2);

  dx = test_alf_dx(lmax);
  for (x = -1.0 + dx; x < 1.0 - dx; x += dx)
    {
      gsl_sf_alf_deriv_array(lmax, mmax, x, p_schmidt, dp_schmidt);
      gsl_sf_alf_deriv_array(lmax, mmax, x, p, dp);

      for (l = 0; l <= lmax; ++l)
        {
          size_t M = GSL_MIN(l, mmax);
          double a_lm = sqrt(2.0 / (double)l / (l + 1.0));
          size_t idx;

          /* test S(l,0) = P(l,0) */
          idx = gsl_sf_alf_array_index(l, 0, lmax);
          gsl_test_rel(p[idx], p_schmidt[idx], tol,
                       "%s l=%zu, m=0, x=%f", desc, l, x);
          gsl_test_rel(dp[idx], dp_schmidt[idx], tol,
                       "%s deriv l=%zu, m=0, x=%f", desc, l, x);

          /* test S(l,m) = a_{lm} * P(l,m) for m > 0 */
          for (m = 1; m <= M; ++m)
            {
              idx = gsl_sf_alf_array_index(l, m, lmax);

              gsl_test_rel(a_lm * p[idx], p_schmidt[idx], tol,
                           "%s l=%zu, m=%zu, x=%f", desc, l, m, x);
              gsl_test_abs(a_lm * dp[idx], dp_schmidt[idx], tol,
                           "%s deriv l=%zu, m=%zu, x=%f", desc, l, m, x);

              a_lm /= sqrt((double) (l + m + 1)) *
                      sqrt((double) (l - m));
            }
        }

      gsl_sf_alf_array(lmax, mmax, x, p2);

      /* test if p = p2 */
      for (m = 0; m <= mmax; ++m)
        {
          for (l = m; l <= lmax; ++l)
            {
              size_t idx = gsl_sf_alf_array_index(l, m, lmax);
              gsl_test_rel(p2[idx], p[idx], tol,
                           "%s compare l=%zu, m=%zu, x=%f",
                           desc, l, m, x);
            }
        }
    }

  free(p);
  free(p2);
  free(dp);
  free(p_schmidt);
  free(dp_schmidt);

  return s;
}

static int
test_alf_all(const size_t lmax, const size_t mmax)
{
  int s = 0;

  s += test_alf_schmidt(lmax, mmax, 0, "schmidt nocsphase");
  s += test_alf_schmidt(lmax, mmax, GSL_SF_ALF_FLG_CSPHASE, "schmidt csphase");

  s += test_alf_norm(GSL_SF_ALF_SPHARM, lmax, mmax, 0, "spharm nocsphase");
  s += test_alf_norm(GSL_SF_ALF_SPHARM, lmax, mmax, GSL_SF_ALF_FLG_CSPHASE, "spharm csphase");

  s += test_alf_norm(GSL_SF_ALF_FULL, lmax, mmax, 0, "full nocsphase");
  s += test_alf_norm(GSL_SF_ALF_FULL, lmax, mmax, GSL_SF_ALF_FLG_CSPHASE, "full csphase");

  s += test_alf_norm(GSL_SF_ALF_FOURPI, lmax, mmax, 0, "fourpi nocsphase");
  s += test_alf_norm(GSL_SF_ALF_FOURPI, lmax, mmax, GSL_SF_ALF_FLG_CSPHASE, "fourpi csphase");

  s += test_alf_unnorm(lmax, mmax, 0, "unnorm nocsphase");
  s += test_alf_unnorm(lmax, mmax, GSL_SF_ALF_FLG_CSPHASE, "unnorm csphase");

  return s;
}

/* test associated legendre functions */
int
test_alf(void)
{
  int s = 0;
  size_t l, m;

  for (l = 0; l <= 10; ++l)
    {
      for (m = 0; m <= l; ++m)
        test_alf_all(l, m);
    }

  test_alf_all(140, 140);
#if 0
  test_alf_all(1000, 1000);
  test_alf_all(2700, 2700);
#endif

  return s;
}
