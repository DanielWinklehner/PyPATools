# PyPATools - Python Particle Accelerator Tools
Several useful 'core' tools for simulations of particle accelerators. This is meant to be the only internal dependency of my other projects.

## OpenCL

For the new bempp-cl it is necessary to have OpenCL drivers installed so that
pyopencl can access the hardware. There are various devices (CPU's, GPU's FPGA's) 
that support OpenCL and it is somewhat up to the user which drivers they install and
which devices they want to use. 

More on the bempp-cl [github page](https://github.com/bempp/bempp-cl)

### OpenCL on Windows with Intel processor
This is my setup (DW), so i downloaded the Intel OpenCL drivers from here:

https://software.intel.com/en-us/articles/opencl-drivers#cpu-section

unfortunately a "free" account is necessary to download the runtime. 

#### Some personal experiences (DW):
I had to uninstall the Intel graphics driver first, which didn't matter because
of the discrete NVIDIA graphics card. 

However, the opencl.dll in the windows/system32 folder is overwritten 
everytime a new graphics driver is installed from NVIDIA. 
Currently I have to remove and reinstall the Intel OpenCL runtime everytime 
I update the graphics driver. 

### OpenCL on Ubuntu 18
TODO

### OpenCL on WSL
Apparently, there is no support for this. i would be happy to learn differently. 
Shoot me an message on github if you know more! -DW