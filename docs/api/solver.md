# `neq_kk_ipt_solver.solver`

The `Solver` class is the main entry point: construct it from a JSON-schema-checked
input dictionary, call {py:meth}`~neq_kk_ipt_solver.Solver.solve`, then read off
{py:attr}`~neq_kk_ipt_solver.Solver.n_occ`, {py:attr}`~neq_kk_ipt_solver.Solver.n_double`,
{py:attr}`~neq_kk_ipt_solver.Solver.GF`, {py:attr}`~neq_kk_ipt_solver.Solver.SE`.

The methods below marked private (leading underscore) are internal, but three
of them -- {py:meth}`~neq_kk_ipt_solver.Solver._calc_band_shift_weiss`,
{py:meth}`~neq_kk_ipt_solver.Solver._calc_band_shift_interacting` and
{py:meth}`~neq_kk_ipt_solver.Solver._build_state` -- carry the actual
equations of the optional Potthoff-Wegner-Nolting band-shift correction and
are exposed here for that reason; see {doc}`../theory/potthoff_band_shift`
for the physics they implement.

```{eval-rst}
.. autoclass:: neq_kk_ipt_solver.Solver
   :members:
   :undoc-members:
   :special-members: __init__
   :private-members: _calc_band_shift_weiss, _calc_band_shift_interacting, _build_state
   :show-inheritance:

.. autoclass:: neq_kk_ipt_solver.ConvergenceError
   :show-inheritance:
```
