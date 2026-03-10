import os
import platform
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
import torch

here = os.path.abspath(os.path.dirname(__file__))

libraries_dir = [os.path.join(os.path.dirname(torch.__file__), 'lib')]
includes_dir = [
        os.path.join(here, '../helper_lib'),
        ##//@@ Add the path to the openCL function library here as stated in the docs
        os.path.join(here, 'opencl_functions')
]

libraries = ['OpenCL', 'm', 'clblast'] 

if platform.system() == 'Linux':
    includes_dir.append('/usr/local/cuda/include')
    libraries_dir.append('/usr/local/cuda/lib64')



setup(
    name='opencl_functions',
    ext_modules=[
        CppExtension(
            name='opencl_functions',
            sources=[
                '../helper_lib/device.c',
                '../helper_lib/kernel.c',
                'opencl_functions/opencl-functions.cpp',
                'opencl_functions/ocl_wrapper_torch.cpp',
            ],
            include_dirs=includes_dir,
            libraries=libraries,
            library_dirs=libraries_dir,
            runtime_library_dirs=[os.path.join(os.path.dirname(torch.__file__), 'lib')], 
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)