import argparse
import random
import ufedmm

import parmed as pmd

from simtk import openmm, unit
from simtk.openmm import app
from sys import stdout, float_info
from ufedmm import cvlib

gb_models = ['HCT', 'OBC1', 'OBC2', 'GBn', 'GBn2']

parser = argparse.ArgumentParser()
parser.add_argument('--case', dest='case', help='the simulation case', default='disarcosine')
parser.add_argument('--implicit-solvent', dest='gb_model', help='an implicit solvent model', choices=gb_models)
parser.add_argument('--salt-molarity', dest='salt_molarity', help='the salt molarity', type=float, default=float_info.min)
parser.add_argument('--seed', dest='seed', help='the RNG seed')
parser.add_argument('--platform', dest='platform', help='the computation platform', default='Reference')
parser.add_argument('--print', dest='print', help='print results?', choices=['yes', 'no'], default='yes')
args = parser.parse_args()

totaltime = 5*unit.nanoseconds
sampling_interval = 200*unit.femtoseconds

temp = 300*unit.kelvin
gamma = 10/unit.picoseconds
dt = 2*unit.femtoseconds
total_time = 10*unit.nanoseconds
mass = 30*unit.dalton*(unit.nanometer/unit.radians)**2
Ks = 1000*unit.kilojoules_per_mole/unit.radians**2
Ts = 1500*unit.kelvin
limit = 180*unit.degrees
sigma = 18*unit.degrees
height = 2.0*unit.kilojoules_per_mole
deposition_period = 200


def dihedral_angle_cvs(prmtop_file, name, *atom_types):
    selected_dihedrals = set()
    for dihedral in pmd.amber.AmberParm(prmtop_file).dihedrals:
        atoms = [getattr(dihedral, f'atom{i+1}') for i in range(4)]
        if all(a.type == atype for a, atype in zip(atoms, atom_types)):
            selected_dihedrals.add(tuple(a.idx for a in atoms))
    n = len(selected_dihedrals)
    collective_variables = []
    for i, dihedral in enumerate(selected_dihedrals):
        force = openmm.CustomTorsionForce('theta')
        force.addTorsion(*dihedral, [])
        cv_name = name if n == 1 else f'{name}{i+1}'
        cv = ufedmm.CollectiveVariable(cv_name, force)
        collective_variables.append(cv)
        # print(cv_name, dihedral)
    return collective_variables


seed = random.SystemRandom().randint(0, 2**31) if args.seed is None else args.seed
nsteps = round(totaltime/dt)
report_interval = round(sampling_interval/dt)

inpcrd = app.AmberInpcrdFile(f'{args.case}.inpcrd')
prmtop = app.AmberPrmtopFile(f'{args.case}.prmtop')

system = prmtop.createSystem(
        nonbondedMethod=app.NoCutoff if prmtop.topology.getNumChains() == 1 else app.PME,
        implicitSolvent=None if args.gb_model is None else getattr(app, args.gb_model),
        implicitSolventSaltConc=args.salt_molarity*unit.moles/unit.liter,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=False,
)

phi_angles = dihedral_angle_cvs(f'{args.case}.prmtop', 'phi', 'c', 'n', 'c3', 'c')
s_phi = ufedmm.DynamicalVariable('s_phi', -limit, limit, mass, Ts, phi_angles[0], Ks, sigma=sigma)

psi_angles = dihedral_angle_cvs(f'{args.case}.prmtop', 'psi', 'n', 'c3', 'c', 'n')
s_psi = ufedmm.DynamicalVariable('s_psi', -limit, limit, mass, Ts, psi_angles[0], Ks, sigma=sigma)

ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], temp, height, deposition_period)
ufedmm.serialize(ufed, 'ufed_object.yml')
integrator = ufedmm.GeodesicLangevinIntegrator(temp, gamma, dt)
integrator.setRandomNumberSeed(seed)
# print(integrator)
platform = openmm.Platform.getPlatformByName(args.platform)
simulation = ufed.simulation(prmtop.topology, system, integrator, platform)
simulation.context.setPositions(inpcrd.positions)
simulation.context.setVelocitiesToTemperature(temp, seed)
output = ufedmm.Tee(stdout, 'output.csv') if args.print == 'yes' else 'output.csv'
reporter = ufedmm.StateDataReporter(output, report_interval, step=True, multipleTemperatures=True, variables=True, speed=True)
simulation.reporters.append(reporter)
simulation.step(nsteps)
