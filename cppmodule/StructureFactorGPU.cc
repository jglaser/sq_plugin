/*! \file StructureFactorGPU.cc
 *  \brief Implements the StructureFactorGPU class
 */
#include "StructureFactorGPU.h"

#ifdef ENABLE_CUDA

#include "StructureFactorGPU.cuh"

StructureFactorGPU::StructureFactorGPU(boost::shared_ptr<SystemDefinition> sysdef,
                          const std::vector<Scalar>& mode,
                          const std::vector<int3>& lattice_vectors,
                          const std::string& filename,
                          bool overwrite)
    : StructureFactor(sysdef, mode, lattice_vectors, filename, overwrite)
    {

    GPUArray<Scalar> gpu_mode(mode.size(), m_exec_conf);
    m_gpu_mode.swap(gpu_mode);

    // Load mode information
    ArrayHandle<Scalar> h_gpu_mode(m_gpu_mode, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < mode.size(); i++)
        h_gpu_mode.data[i] = mode[i];

    m_block_size = 512;
    unsigned int max_n_blocks = m_pdata->getMaxN()/m_block_size + 1;

    GPUArray<Scalar2> fourier_mode_scratch(mode.size()*max_n_blocks, m_exec_conf);
    m_fourier_mode_scratch.swap(fourier_mode_scratch);

    m_wave_vectors_updated = 0;
    }

void StructureFactorGPU::computeFourierModes(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "Structure factor");

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);

    if (m_fourier_mode_scratch.getNumElements() != m_pdata->getMaxN())
        {
        unsigned int max_n_blocks = m_pdata->getMaxN()/m_block_size + 1;
        m_fourier_mode_scratch.resize(max_n_blocks*m_fourier_modes.getNumElements());
        }

        {
        ArrayHandle<Scalar3> d_wave_vectors(m_wave_vectors, access_location::device, access_mode::read);
        ArrayHandle<Scalar2> d_fourier_modes(m_fourier_modes, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_gpu_mode(m_gpu_mode, access_location::device, access_mode::read);
        ArrayHandle<Scalar2> d_fourier_mode_scratch(m_fourier_mode_scratch, access_location::device, access_mode::overwrite);

        // calculate Fourier modes
        gpu_calculate_fourier_modes(m_wave_vectors.getNumElements(),
                                    d_wave_vectors.data,
                                    m_pdata->getN(),
                                    d_postype.data,
                                    d_gpu_mode.data,
                                    d_fourier_modes.data,
                                    m_block_size,
                                    d_fourier_mode_scratch.data
                                    );

        CHECK_CUDA_ERROR();
        }

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }


void export_StructureFactorGPU()
    {
    class_<StructureFactorGPU, boost::shared_ptr<StructureFactorGPU>, bases<StructureFactor>, boost::noncopyable >
        ("StructureFactorGPU", init< boost::shared_ptr<SystemDefinition>,
                                         const std::vector<Scalar>&,
                                         const std::vector<int3>&,
                                         const std::string&,
                                         bool>());
    }
#endif
