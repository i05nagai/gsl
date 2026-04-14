/* rstat/test_wrstat.c
 * 
 * Copyright (C) 2026 Patrick Alken
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
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_wrstat.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_test.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

static void
test_wrstat_basic(const double tol, const size_t n, const double weights[], const double data[],
                  const char * desc)
{
  gsl_wrstat_workspace *wrstat_workspace_p = gsl_wrstat_alloc();
  const double expected_wmean = gsl_stats_wmean(weights, 1, data, 1, n);
  double expected_rms = 0.0;
  double wmean, var, sd, rms, norm;
  double sumW = 0.0;
  size_t i, num;
  int status;

  /* compute expected rms */
  for (i = 0; i < n; ++i)
    {
      expected_rms += weights[i] * data[i] * data[i];
      sumW += weights[i];
    }

  expected_rms = sqrt(expected_rms / sumW);

  /* add data to wrstat workspace */
  for (i = 0; i < n; ++i)
    gsl_wrstat_add(weights[i], data[i], wrstat_workspace_p);

  wmean    = gsl_wrstat_mean(wrstat_workspace_p);
  rms      = gsl_wrstat_rms(wrstat_workspace_p);
  num      = gsl_wrstat_n(wrstat_workspace_p);

  gsl_test_int(num, n, "%s n n=%zu", desc, n);
  gsl_test_rel(wmean, expected_wmean, tol, "%s wmean n=%zu", desc, n);
  gsl_test_rel(rms, expected_rms, tol, "%s wrms n=%zu", desc, n);
  
  if (n > 1)
    {
      const double expected_var = gsl_stats_wvariance(weights, 1, data, 1, n);
      const double expected_sd = gsl_stats_wsd(weights, 1, data, 1, n);

      var      = gsl_wrstat_variance(wrstat_workspace_p);
      sd       = gsl_wrstat_sd(wrstat_workspace_p);

      gsl_test_rel(var, expected_var, tol, "%s wvariance n=%zu", desc, n);
      gsl_test_rel(sd, expected_sd, tol, "%s wstddev n=%zu", desc, n);
    }

  status = gsl_wrstat_reset(wrstat_workspace_p);
  gsl_test_int(status, GSL_SUCCESS, "%s: wrstat returned success", desc);
  num = gsl_wrstat_n(wrstat_workspace_p);

  gsl_test_int(num, 0, "n n=%zu" , n);

  gsl_wrstat_free(wrstat_workspace_p);
}

static int
test_wrstat(gsl_rng * r)
{
  const double tol1 = 1.0e-8;
  const double tol2 = 1.0e-3;

  {
    const size_t N = 2000000;
    double *data = malloc(N * sizeof(double));
    double data2[5], weights2[5];
    size_t i, j;
    char buf[64];

    /* test1: test on small datasets n <= 5 (median will be exact in this case) */
    for (i = 0; i < 100; ++i)
      {
        random_data(-1.0, 1.0, 5, data2, r);
        random_data(0.0, 10.0, 5, weights2, r);

        for (j = 1; j <= 5; ++j)
          {
            sprintf(buf, "wrstat test1[%zu,%zu]", i, j);
            test_wrstat_basic(tol1, j, weights2, data2, buf);
          }
      }

#if 0
    /* test2: test on large datasets */

    random_data(-1.0, 1.0, N, data, r);

    for (i = 1; i <= 10; ++i)
      test_wrstat_basic(i, data, tol1, "test2");

    test_wrstat_basic(100, data, tol1, "test2");
    test_wrstat_basic(1000, data, tol1, "test2");
    test_wrstat_basic(10000, data, tol1, "test2");
    test_wrstat_basic(50000, data, tol1, "test2");
    test_wrstat_basic(80000, data, tol1, "test2");
    test_wrstat_basic(1500000, data, tol1, "test2");
    test_wrstat_basic(2000000, data, tol1, "test2");

    /* test3: add large constant */

    for (i = 0; i < 5; ++i)
      data2[i] += 1.0e9;

    test_wrstat_basic(5, data2, 1.0e-6, "test3");
#endif

    free(data);
  }

#if 0
  {
    size_t n = 1000000;
    double *data = malloc(n * sizeof(double));
    double *sorted_data = malloc(n * sizeof(double));
    gsl_rstat_workspace *rstat_workspace_p = gsl_rstat_alloc();
    double p;
    size_t i;

    for (i = 0; i < n; ++i)
      {
        data[i] = gsl_ran_gaussian_tail(r, 1.3, 1.0);
        gsl_rstat_add(data[i], rstat_workspace_p);
      }

    memcpy(sorted_data, data, n * sizeof(double));
    gsl_sort(sorted_data, 1, n);

    /* test mean, variance */
    {
      const double expected_mean = gsl_stats_mean(data, 1, n);
      const double expected_var = gsl_stats_variance(data, 1, n);
      const double expected_sd = gsl_stats_sd(data, 1, n);
      const double expected_skew = gsl_stats_skew(data, 1, n);
      const double expected_kurtosis = gsl_stats_kurtosis(data, 1, n);
      const double expected_median = gsl_stats_quantile_from_sorted_data(sorted_data, 1, n, 0.5);

      const double mean = gsl_rstat_mean(rstat_workspace_p);
      const double var = gsl_rstat_variance(rstat_workspace_p);
      const double sd = gsl_rstat_sd(rstat_workspace_p);
      const double skew = gsl_rstat_skew(rstat_workspace_p);
      const double kurtosis = gsl_rstat_kurtosis(rstat_workspace_p);
      const double median = gsl_rstat_median(rstat_workspace_p);

      gsl_test_rel(mean, expected_mean, tol1, "mean");
      gsl_test_rel(var, expected_var, tol1, "variance");
      gsl_test_rel(sd, expected_sd, tol1, "stddev");
      gsl_test_rel(skew, expected_skew, tol1, "skew");
      gsl_test_rel(kurtosis, expected_kurtosis, tol1, "kurtosis");
      gsl_test_abs(median, expected_median, tol2, "median");
    }

    free(data);
    free(sorted_data);
    gsl_rstat_free(rstat_workspace_p);
  }
#endif

  return GSL_SUCCESS;
}
