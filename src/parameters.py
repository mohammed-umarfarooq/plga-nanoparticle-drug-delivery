# =====================================================
# DOMAIN SIZE
# =====================================================

Lx = 10e-3      # 10 mm
Ly = 10e-3      # 10 mm

# =====================================================
# GRID
# =====================================================

Nx = 50
Ny = 50

dx = Lx / Nx
dy = Ly / Ny

# =====================================================
# TRANSPORT PARAMETERS
# =====================================================

# Base diffusion coefficient
# Particle-size study will modify this using
# the Stokes-Einstein equation

D = 5e-8

# Hydraulic conductivity
# Adjusted to 5e-13 for realistic physiological transport

K = 5e-13

# Cellular uptake coefficient
# Adjusted to match transport model updates

uptake_rate = 0.015

# =====================================================
# DRUG RELEASE
# =====================================================

release_rate = 0.05

# =====================================================
# TUMOR GROWTH
# =====================================================

growth_rate = 0.025

# Drug efficacy

drug_efficacy = 0.85

# Carrying capacity

carrying_capacity = 2e7

# =====================================================
# TIME PARAMETERS
# =====================================================

dt = 0.1

time_steps = 1000

# =====================================================
# PARTICLE SIZE STUDY
# =====================================================

particle_sizes = [
    50,
    100,
    200
]

# =====================================================
# RELEASE RATE STUDY
# =====================================================

release_rates = {

    "Slow": 0.02,

    "Medium": 0.05,

    "Fast": 0.10
}