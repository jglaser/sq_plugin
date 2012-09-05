from hoomd_plugins.sq_plugin import _sq_plugin

# Next, since we are extending an updater, we need to bring in the base class updater and some other parts from 
# hoomd_script
from hoomd_script.analyze import _analyzer
from hoomd_script import util
from hoomd_script import globals
import hoomd

class sq(_analyzer):
    def __init__(self, filename, mode, lattice_vectors, period=1, overwrite=False):
        util.print_status_line();
    
        # initialize base class
        _analyzer.__init__(self);
       
        if len(lattice_vectors) == 0:
                globals.msg.error("analyze.sq: List of supplied latice vectors is empty.\n")
                raise RuntimeEror('Error creating collective variable.')

        if type(mode) != type(dict()):
                globals.msg.error("analyze.sq: Mode amplitudes specified incorrectly.\n")
                raise RuntimeEror('Error creating collective variable.')

        cpp_mode = hoomd.std_vector_float()
        for i in range(0, globals.system_definition.getParticleData().getNTypes()):
            t = globals.system_definition.getParticleData().getNameByType(i)

            if t not in mode.keys():
                globals.msg.error("cv.lamellar: Missing mode amplitude for particle type " + t + ".\n")
                raise RuntimeEror('Error creating collective variable.')
            cpp_mode.append(mode[t])

        cpp_lattice_vectors = _sq_plugin.std_vector_int3()
        for l in lattice_vectors:
            if len(l) != 3:
                globals.msg.error("cv.lamellar: List of input lattice vectors not a list of triples.\n")
                raise RuntimeError('Error creating collective variable.')
            cpp_lattice_vectors.append(hoomd.make_int3(l[0], l[1], l[2]))

        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_analyzer = _sq_plugin.StructureFactor(globals.system_definition, cpp_mode, cpp_lattice_vectors, filename, overwrite);
        else:
            self.cpp_analyzer = _sq_plugin.StructureFactorGPU(globals.system_definition, cpp_mode, cpp_lattice_vectors, filename, overwrite);

        self.setupAnalyzer(period)
