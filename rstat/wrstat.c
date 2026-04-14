/* rstat/wrstat.c
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
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_wrstat.h>

gsl_wrstat_workspace *
gsl_wrstat_alloc(void)
{
  gsl_wrstat_workspace *w;

  w = calloc(1, sizeof(gsl_wrstat_workspace));

  if (w == 0)
    {
      GSL_ERROR_NULL ("failed to allocate space for workspace", GSL_ENOMEM);
    }

  gsl_wrstat_reset(w);

  return w;
}

void
gsl_wrstat_free(gsl_wrstat_workspace *w)
{
  free(w);
}

size_t
gsl_wrstat_n(const gsl_wrstat_workspace *w)
{
  return w->n;
}

/* add a data point to the running totals */
int
gsl_wrstat_add(const double wt, const double x, gsl_wrstat_workspace * w)
{
  if (wt < 0.0)
    {
      GSL_ERROR ("weight must be >= 0", GSL_EDOM);
    }
  else
    {
      double delta = x - w->wmean;

      /* update min and max */
      if (w->n == 0)
        {
          w->min = x;
          w->max = x;
        }
      else
        {
          if (x < w->min)
            w->min = x;
          if (x > w->max)
            w->max = x;
        }

      /* update mean and variance */
      if (wt > 0.0)
        {
          double prevW = w->W;
          double ratio;

          w->W += wt;
          w->W2 += wt * wt;

          ratio = wt / w->W;
          w->wmean += delta * ratio;
          w->M2 += prevW * ratio * delta * delta;
        }

      ++(w->n);

      return GSL_SUCCESS;
    }
}

double
gsl_wrstat_min(const gsl_wrstat_workspace * w)
{
  return w->min;
}

double
gsl_wrstat_max(const gsl_wrstat_workspace * w)
{
  return w->max;
}

double
gsl_wrstat_mean(const gsl_wrstat_workspace * w)
{
  return w->wmean;
}

double
gsl_wrstat_variance(const gsl_wrstat_workspace * w)
{
  if (w->n > 1)
    {
      double Wsq = w->W * w->W;
      return (w->W * w->M2 / (Wsq - w->W2));
    }
  else
    return 0.0;
}

double
gsl_wrstat_sd(const gsl_wrstat_workspace * w)
{
  double var = gsl_wrstat_variance(w);
  return (sqrt(var));
}

double
gsl_wrstat_rms(const gsl_wrstat_workspace * w)
{
  double rms = 0.0;

  if (w->n > 0)
    {
      double wmean = gsl_wrstat_mean(w);
      rms = gsl_hypot(wmean, sqrt(w->M2 / w->W));
    }

  return rms;
}

int
gsl_wrstat_reset(gsl_wrstat_workspace * w)
{
  w->W = 0.0;
  w->W2 = 0.0;
  w->min = 0.0;
  w->max = 0.0;
  w->wmean = 0.0;
  w->M2 = 0.0;
  w->n = 0;

  return GSL_SUCCESS;
}
