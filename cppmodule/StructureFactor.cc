/* \file StructureFactor.cc
 * \brief Implements the StructureFactor class
 */
#include "StructureFactor.h"

#include <iomanip>

#include <boost/python.hpp>
#include <boost/filesystem.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;
using namespace boost::filesystem;

StructureFactor::StructureFactor(boost::shared_ptr<SystemDefinition> sysdef,
                               const std::vector<Scalar>& mode,
                               const std::vector<int3>& lattice_vectors,
                               const std::string& filename,
                               bool overwrite)
    : Analyzer(sysdef), m_lattice_vectors(lattice_vectors), m_mode(mode)
    {
    if (mode.size() != m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "cv.lamellar: Number of mode parameters has to equal the number of particle types!" << std::endl;
        throw runtime_error("Error initializing cv.lamellar");
        }

    // allocate array of wave vectors
    GPUArray<Scalar3> wave_vectors(m_lattice_vectors.size(), m_exec_conf);
    m_wave_vectors.swap(wave_vectors);

    GPUArray<Scalar2> fourier_modes(m_lattice_vectors.size(), m_exec_conf);
    m_fourier_modes.swap(fourier_modes);

    m_filename = filename;
    m_delimiter = "\t";
    m_appending = !overwrite;
    openOutputFiles();
    }

void StructureFactor::openOutputFiles()
    {
#ifdef ENABLE_MPI
    // only output to file on root processor
    if (m_pdata->getDomainDecomposition())
        if (! m_exec_conf->isRoot())
            return;
#endif
    // open the file
    if (exists(m_filename) && m_appending)
        {
        m_exec_conf->msg->notice(3) << "sq-plugin.analyze.sq: Appending log to existing file \"" << m_filename << "\"" << endl;
        m_file.open(m_filename.c_str(), ios_base::in | ios_base::out | ios_base::ate);
        }
    else
        {
        m_exec_conf->msg->notice(3) << "sq-plugin.analzye.sq: Creating new log in file \"" << m_filename << "\"" << endl;
        m_file.open(m_filename.c_str(), ios_base::out);
        m_appending = false;
        }

    if (!m_file.good())
        {
        m_exec_conf->msg->error() << "sq-plugin.analzye.sq: Error opening log file " << m_filename << endl;
        throw runtime_error("Error initializing Logger");
        }
    }

void StructureFactor::analyze(unsigned int timestep)
    {
    calculateWaveVectors();

    this->computeFourierModes(timestep);

    ArrayHandle<Scalar2> h_fourier_modes(m_fourier_modes, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_wave_vectors(m_wave_vectors, access_location::host, access_mode::read);

    Scalar3 L = m_pdata->getGlobalBox().getL();
    Scalar V = L.x*L.y*L.z;

    bool is_root = true;
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        is_root = m_exec_conf->isRoot();
#endif

    // Calculate value of collective variable (sum of real parts of fourier modes)
    for (unsigned k = 0; k < m_fourier_modes.getNumElements(); k++)
        {
        Scalar re = Scalar(0.0);
        Scalar im = Scalar(0.0);

#ifdef ENABLE_MPI
        // reduce value of fourier mode on root processor
        if (m_pdata->getDomainDecomposition())
            {
            MPI_Reduce(&h_fourier_modes.data[k].x,&re,1, MPI_HOOMD_SCALAR, MPI_SUM, 0, m_exec_conf->getMPICommunicator());
            MPI_Reduce(&h_fourier_modes.data[k].y,&im,1, MPI_HOOMD_SCALAR, MPI_SUM, 0, m_exec_conf->getMPICommunicator());
            }
        else
#endif
            {
            re = h_fourier_modes.data[k].x;
            im = h_fourier_modes.data[k].y;
            }

        if (is_root)
            {
            // write to output file
            
            // write timestep
            m_file << setprecision(10) << timestep << m_delimiter;

            // write Miller indices
            m_file << setprecision(10) << m_lattice_vectors[k].x << m_delimiter;
            m_file << setprecision(10) << m_lattice_vectors[k].y << m_delimiter;
            m_file << setprecision(10) << m_lattice_vectors[k].z << m_delimiter;

            // write norm of wave vector
            Scalar3 q = h_wave_vectors.data[k];
            Scalar norm = sqrt(dot(q,q));
            m_file << setprecision(10) << norm << m_delimiter;

            // write fourier mode and structure factor
            m_file << setprecision(10) << re << m_delimiter;
            m_file << setprecision(10) << im << m_delimiter;

            Scalar sq = Scalar(1.0)/V*(re*re+im*im);

            m_file << setprecision(10) << sq;
            m_file << std::endl;
            }
        } 

    }


//! Calculate wave vectors
void StructureFactor::calculateWaveVectors()
    {
    ArrayHandle<Scalar3> h_wave_vectors(m_wave_vectors, access_location::host, access_mode::overwrite);

    const BoxDim &box = m_pdata->getGlobalBox();
    const Scalar3 L = box.getL();

    for (unsigned int k = 0; k < m_lattice_vectors.size(); k++)
        h_wave_vectors.data[k] = 2*M_PI*make_scalar3(m_lattice_vectors[k].x/L.x,
                                              m_lattice_vectors[k].y/L.y,
                                              m_lattice_vectors[k].z/L.z);
    }

//! Returns a list of fourier modes (for all wave vectors)
void StructureFactor::computeFourierModes(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("Structure Factor");

    ArrayHandle<Scalar2> h_fourier_modes(m_fourier_modes, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar3> h_wave_vectors(m_wave_vectors, access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);

    for (unsigned int k = 0; k < m_wave_vectors.getNumElements(); k++)
        {
        h_fourier_modes.data[k] = make_scalar2(0.0,0.0);
        Scalar3 q = h_wave_vectors.data[k];
        
        for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
            {
            Scalar4 postype = h_postype.data[idx];

            Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
            unsigned int type = __scalar_as_int(postype.w);
            Scalar mode = m_mode[type];
            Scalar dotproduct = dot(q,pos);
            h_fourier_modes.data[k].x += mode * cos(dotproduct);
            h_fourier_modes.data[k].y += mode * sin(dotproduct);
            }
        }

    if (m_prof) m_prof->pop();
    }

void export_StructureFactor()
    {
    class_<StructureFactor, boost::shared_ptr<StructureFactor>, bases<Analyzer>, boost::noncopyable >
        ("StructureFactor", init< boost::shared_ptr<SystemDefinition>,
                                         const std::vector<Scalar>&,
                                         const std::vector<int3>,
                                         const std::string&,
                                         bool>());

    class_<std::vector<int3> >("std_vector_int3")
        .def(vector_indexing_suite< std::vector<int3> > ())
        ;
    }
