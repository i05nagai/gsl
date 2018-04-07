/* movstat/movqqr.c
 *
 * Compute moving q-quantile range
 * 
 * Copyright (C) 2018 Patrick Alken
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
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_movstat.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>

#include "movstat_common.c"

/*
gsl_movstat_qqr()
  Apply a moving q-quantile range to an input vector

Inputs: endtype - how to handle end points
        x       - input vector, size n
        q       - quantile \in [0,0.5]
        xqqr    - (output) vector of q-quantile ranges of x, size n
                  xqqr_i = q-quantile range of i-th window:
        w       - workspace
*/

int
gsl_movstat_qqr(const gsl_movstat_end_t endtype, const gsl_vector * x, const double q,
                gsl_vector * xqqr, gsl_movstat_workspace * w)
{
  if (x->size != xqqr->size)
    {
      GSL_ERROR("x and xqqr vectors must have same length", GSL_EBADLEN);
    }
  else if (q < 0.0 || q > 0.5)
    {
      GSL_ERROR("q must be between 0 and 0.5", GSL_EDOM);
    }
  else
    {
      const int n = (int) x->size;
      const int H = (int) w->H; /* number of samples to left of current sample */
      const int J = (int) w->J; /* number of samples to right of current sample */
      double *window = w->work;
      int window_size, i;

      for (i = 0; i < n; ++i)
        {
          double *xqqri = gsl_vector_ptr(xqqr, i);
          double quant1, quant2;

          /* fill window centered on x_i */
          window_size = movstat_fill_window(endtype, i, H, J, x, window);

          /* sort window */
          gsl_sort(window, 1, window_size);

          /* compute q-quantile and (1-q)-quantile */
          quant1 = gsl_stats_quantile_from_sorted_data(window, 1, window_size, q);
          quant2 = gsl_stats_quantile_from_sorted_data(window, 1, window_size, 1.0 - q);

          /* compute q-quantile range */
          *xqqri = quant2 - quant1;
        }

      return GSL_SUCCESS;
    }
}