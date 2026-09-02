from .simulation_paras import *
from ..helpers import derivative


def physical_residual(
    u,
    x,
    y,
    t,
    heat_diffusivity_x=HEAT_DIFFUSIVITY_X,
    heat_diffusivity_y=HEAT_DIFFUSIVITY_Y,
):
    x.grad = None
    y.grad = None
    t.grad = None
    u_t = derivative(u, t, order=1)
    u_x = derivative(u, x, order=1)
    u_y = derivative(u, y, order=1)
    u_xx = derivative(u_x, x, order=1)
    u_yy = derivative(u_y, y, order=1)
    return u_t - heat_diffusivity_x * u_xx - heat_diffusivity_y * u_yy
