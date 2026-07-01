import os
import sys

print('WORKDIR', os.getcwd())
print('PYTHON', sys.executable)

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from solver import run_simulation

print('START RUN')
run_simulation()
print('END RUN')
