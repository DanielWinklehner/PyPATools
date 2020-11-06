conda deactivate
conda env remove --name PyPATools
cd ~/src/PyPATools/PyPATools/documents/
conda env create --file env_ubuntu.yml
conda activate pyPATools
cd ~/src/pygist/
python setup.py config
python setup.py install
cd ~/src/warp/pywarp90/
rm -rf build3
make install
cd ~/src/warp/warp_test/
python runalltests.py
