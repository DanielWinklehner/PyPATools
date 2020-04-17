# PyPATools - Python Particle Accelerator Tools
Several useful 'core' tools for simulations of particle accelerators. 
This is meant to be the only internal dependency of all repositories
in the Python Particle Accelerator Tools project.

## The Settings.txt file
During installation a Settings.txt file is created in
the same directory that settings.py is being installed to 
(typically in _.../site-packages/PyPATools/PyPATools/_). The settings handler 
will look for this Settings.txt file every time the script is loaded. 
The user can change settings in this file. Better would be to copy it to a 
local directory.

In Windows, this should be:

``$APPDATA\PyPATools\Settings.txt``

(typically this is _C:\Users\<Username>\AppData\Roaming\PyPATools_). 
The _PyPATools_ subdirectory has to be created manually.

In Linux it should be:

``$HOME/.local/PyPATools/Settings.txt``

The settings handler wil look in those directories first and in the package path 
second. 

_Note: The py_electrodes module comes with its own Settings.txt file that is set 
up in the same way, but lives in the py_electrodes/ subdirectory_

## Setting up the Anaconda3 environment

With the new version of OpenCascade (OCC) and 
[pythonocc-core](https://github.com/tpaviot/pythonocc-core) (7.4.0), a simple 
anaconda installation from yml file is possible. The file _environment.yml_ 
can be found in /PyPATools/documents. 

simply create a new Anaconda3 environment from the Navigator (Environments-->Import)
or from the command line

``conda env create -f environment.yml``

The environment name can be changed in the yml file or with the _-name_ flag.

_Note: At the moment, this yml file installs EVERYTHING, including 
several dependencies for the other
repositories in the Python Particle Accelerator Tools project. 
We will provide a minimal installation for just the scripts in PyPATools soon._ 

### Additional notes for Windows

A community edition of MS Visual Studio (2017 or newer) is also needed to compile 
some of the c-extensions for cython. 
Get it for free here: https://visualstudio.microsoft.com/downloads/

### Additional notes for Ubuntu 18 (WSL and local installation)
In most cases the c compilers for cython are included in the Ubuntu installation.
If not, the build-essential package should have what is needed.

``sudo apt-get install build-essential``

It looks like the 3D rendering drivers are missing in a vanilla Ubuntu 18 installation, 
install them using:

``sudo apt-get install libglu1-mesa``
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