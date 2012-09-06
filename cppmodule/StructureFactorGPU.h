/*! \file StructureFactorGPU.h
 *  \brief Defines the StructureFactorGPU class
 */

#ifndef __STRUCUTRE_FACTOR_GPU_H__
#define __STRUCUTRE_FACTOR_GPU_H__

#include "StructureFactor.h"

#ifdef ENABLE_CUDA

//! Class to calculate the lamellar order parameter on the GPU
class StructureFactorGPU : public StructureFactor
    {
    public:
        StructureFactorGPU(boost::shared_ptr<SystemDefinition> sysdef,
                               const std::vector<Scalar>& mode,
                               const std::vector<int3>& lattice_vectors,
                               const std::string& filename,
                               bool overwrite);

        virtual ~StructureFactorGPU() {}

    protected:
        // calculates current CV value
        virtual void computeFourierModes(unsigned int timestep);

    private:
        GPUArray<Scalar> m_gpu_mode;       //!< Factors multiplying per-type densities to obtain scalar quantity
        unsigned int m_wave_vectors_updated; //!< Timestep wave vectors were last updated
        unsigned int m_block_size;          //!< Block size for fourier mode calculation
        GPUArray<Scalar2> m_fourier_mode_scratch; //!< Scratch memory for fourier mode calculation
    };

void export_StructureFactorGPU();
#endif
#endif // __STRUCUTRE_FACTOR_GPU_H__
