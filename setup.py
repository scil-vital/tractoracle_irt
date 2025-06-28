from setuptools import setup
from torch.utils import cpp_extension

setup(
    packages=['tractoracle_irt'],

    # List C++ extensions
    ext_modules=[
        cpp_extension.CppExtension(
            'tractoracle_irt.algorithms.shared.disc_cumsum',
            ['tractoracle_irt/algorithms/shared/disc_cumsum.cpp'],
        ),
    ],

    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
