/* rstat/test.c
 * 
 * Copyright (C) 2015, 2016, 2017, 2018, 2019, 2020, 2021 Patrick Alken
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
#include <gsl/gsl_rstat.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_test.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_ieee_utils.h>

/* generate random data on [a,b] */
int
random_data(const double a, const double b, const size_t n, double data[], gsl_rng *r)
{
  size_t i;

  for (i = 0; i < n; ++i)
    data[i] = (b - a) * gsl_rng_uniform(r) + a;

  return 0;
}

#include "test_rstat.c"
#include "test_wrstat.c"

int
main()
{
  gsl_rng *r = gsl_rng_alloc(gsl_rng_default);

  gsl_ieee_env_setup();

  test_rstat(r);
  test_wrstat(r);

  gsl_rng_free(r);

  exit (gsl_test_summary());
}
