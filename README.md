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

## Setting up the Anaconda3 environment (tested in Windows 11 and WSL2/Ubuntu 22.04)

The simplest setup is using conda (Anaconda3-tested) The file _conda-env.yml_ 
can be found in /PyPATools/documents.

Simply create a new Anaconda3 environment from the Navigator (Environments-->Import)
or from the command line

```bash
conda env create -f conda-env.yml
```

The environment name can be changed in the yml file or with the _--name_ flag.

### Install PyPATools

_Note: there is a co-dependency between PyPATools and py_electrodes. Please install both for full functionality._

Either:

```bash
git clone https://github.com/DanielWinklehner/PyPATools.git
cd PyPATools
pip install -e .

git clone https://github.com/DanielWinklehner/py_electrodes.git
cd py_electrodes
pip install -e .
```

Or directly (if you don't need the source files and examples):

```bash
pip install git+https://github.com/DanielWinklehner/PyPATools.git
pip install git+https://github.com/DanielWinklehner/py_electrodes.git
```

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

## Installing WARP

Installation of WARP follows largely the instructions on the WARP 
[webpage](http://warp.lbl.gov/home/how-to-s/installation).

To use the `warp.sh` bash script in the documents directory, the `src`
directory has to exist in your home folder and both pygist and WARP
have to be cloned there. 

### pygist

Get pygist by cloning it from the repository

```bash
cd ~/src
git clone https://bitbucket.org/dpgrote/pygist
```

### Downloading WARP
Get WARP by downloading it from https://bitbucket.org/berkeleylab/warp/downloads/ and extracting 
it using `tar`:

```bash
cd ~/src
wget https://bitbucket.org/berkeleylab/warp/downloads/Warp_Release_4.5.tgz
tar -xvf Warp_Release_4.5.tgz
cd warp
git pull
```

### Compiling gist and WARP using the shell script
Cd into the documents directory where warp.sh is located.

```bash
source warp.sh
```
The installation should run automatically for gist, serial WARP and parallel WARP.
