# cuFFTMp includes both the single-GPU cuFFT API and the distributed API.
# Use matching cuFFTMp, NVSHMEM and MPI installations from the NVIDIA HPC SDK.
# Roots may be supplied as CMake cache entries or via CUFFTMP_HOME/NVSHMEM_HOME.
# The MPI bootstrap shared library must also be on LD_LIBRARY_PATH at runtime.

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "This cuFFTMp-enabled branch requires Linux.")
endif()

set(CUFFTMP_ROOT "" CACHE PATH "cuFFTMp prefix containing include/ and lib/ or lib64/")
set(NVSHMEM_ROOT "" CACHE PATH "NVSHMEM prefix containing lib/ or lib64/")

find_path(CUFFTMP_INCLUDE_DIR
    NAMES cufftMp.h
    HINTS "${CUFFTMP_ROOT}" "$ENV{CUFFTMP_ROOT}" "$ENV{CUFFTMP_HOME}"
    PATH_SUFFIXES include
)

# cuFFTMp requires the shared NVSHMEM host library, not static NVSHMEM.
set(_cuFFTMp_saved_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")
find_library(CUFFTMP_LIBRARY
    NAMES cufftMp
    HINTS "${CUFFTMP_ROOT}" "$ENV{CUFFTMP_ROOT}" "$ENV{CUFFTMP_HOME}"
    PATH_SUFFIXES lib lib64
)
find_library(NVSHMEM_HOST_LIBRARY
    NAMES nvshmem_host
    HINTS "${NVSHMEM_ROOT}" "$ENV{NVSHMEM_ROOT}" "$ENV{NVSHMEM_HOME}"
    PATH_SUFFIXES lib lib64
)
set(CMAKE_FIND_LIBRARY_SUFFIXES "${_cuFFTMp_saved_suffixes}")
unset(_cuFFTMp_saved_suffixes)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuFFTMp
    REQUIRED_VARS CUFFTMP_INCLUDE_DIR CUFFTMP_LIBRARY NVSHMEM_HOST_LIBRARY
    REASON_FAILURE_MESSAGE
        "Set CUFFTMP_ROOT and NVSHMEM_ROOT to matching HPC SDK component prefixes, or set CUFFTMP_HOME and NVSHMEM_HOME in the environment."
)

if(cuFFTMp_FOUND)
    foreach(_cuFFTMp_library IN ITEMS CUFFTMP_LIBRARY NVSHMEM_HOST_LIBRARY)
        if("${${_cuFFTMp_library}}" MATCHES "\\.a$")
            message(FATAL_ERROR "${_cuFFTMp_library} must name a shared library, not a static archive.")
        endif()
    endforeach()
    unset(_cuFFTMp_library)

    if(NOT TARGET NVSHMEM::host)
        add_library(NVSHMEM::host SHARED IMPORTED GLOBAL)
        set_target_properties(NVSHMEM::host PROPERTIES
            IMPORTED_LOCATION "${NVSHMEM_HOST_LIBRARY}"
        )
    endif()

    if(NOT TARGET cuFFTMp::cuFFTMp)
        add_library(cuFFTMp::cuFFTMp SHARED IMPORTED GLOBAL)
        set_target_properties(cuFFTMp::cuFFTMp PROPERTIES
            IMPORTED_LOCATION "${CUFFTMP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUFFTMP_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "NVSHMEM::host;CUDA::cuda_driver"
        )
    endif()
endif()

mark_as_advanced(CUFFTMP_INCLUDE_DIR CUFFTMP_LIBRARY NVSHMEM_HOST_LIBRARY)
