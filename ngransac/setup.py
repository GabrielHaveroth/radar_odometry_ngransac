from xml.etree.ElementInclude import include
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
# Directory containing OpenCV library files
opencv_lib_dir = '/usr/lib/'  
# Directory containing OpenCV header files
opencv_inc_dir = '/usr/include/opencv2'

setup(
	name='ngransac',
	ext_modules=[CppExtension(
		name='ngransac',
		sources=['ngransac.cpp', 'thread_rand.cpp'],
		include_dirs=[opencv_inc_dir, '/usr/include/eigen3/'],
		library_dirs=[opencv_lib_dir],
		libraries=['opencv_core', 'opencv_calib3d'],
		extra_compile_args=['-fopenmp']
        )],
	cmdclass={'build_ext': BuildExtension})
