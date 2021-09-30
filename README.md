# FHEML        self.nested_timer.start()
        self.encrypted_w = self.encrypt_data([np_to_list(self.w)])[0]
        self.nested_timer.finish()
        self.time_tracking[self.WENC] = self.nested_timer.get_time_in(Timer.TIMEFORMAT_MS)

        self.nested_timer.start()
        self.encrypted_b = self.encrypt_data([[self.b] * self.n_features])[0] # Must be size of data
        self.nested_timer.finish()
        self.time_tracking[self.BENC] = self.nested_timer.get_time_in(Timer.TIMEFORMAT_MS)        self.nested_timer.start()
        self.encrypted_w = self.encrypt_data([np_to_list(self.w)])[0]
        self.nested_timer.finish()
        self.time_tracking[self.WENC] = self.nested_timer.get_time_in(Timer.TIMEFORMAT_MS)

        self.nested_timer.start()
        self.encrypted_b = self.encrypt_data([[self.b] * self.n_features])[0] # Must be size of data
        self.nested_timer.finish()
        self.time_tracking[self.BENC] = self.nested_timer.get_time_in(Timer.TIMEFORMAT_MS)
This project is part of my Bachelor thesis at the School for Electrical Engineering @ Belgrade. 

This project is supposed to analyse the performance of different Python libraries that use the base C++ encryption libraries behind the curtain. 

Libraries that are used are Pyfhel, Palisade & SEAL. Pyfhel uses Palisade or SEAL for its driving mechanism (configurable) whilst the Python Wrapper & SEAL wrapper use their respective libraries.
The idea is to measure and analyse the performance of each library in comparison to eachother.

## System information
The system that this project is run on is:
- OS: PopOS! 21.04 with Nvidia Drivers.
- CPU: AMD Ryzen 5900x
- RAM: DDR4 32GB 3600 CL16
- SSD: Samsung EVO 970 Plus 

# Steps to configure the environment/project
Firstly, install git.

    sudo apt-get install git
After that, go to whichever folder you wish. In that folder we will store all necessary dependencies including our own project. I am using Desktop.

    cd ~/Desktop
  After that, it's necessary to clone the next three projects:
  

    git clone https://gitlab.com/palisade/palisade-development.git
    git clone https://gitlab.com/palisade/palisade-python-demo.git
    git clone https://github.com/Huelse/SEAL-Python.git
After that, we are going to install packages that are needed for compiling the cloned projects

    sudo apt-get install build-essential
    sudo apt-get install autoconf
    sudo apt-get install make
    sudo apt-get install cmake
    sudo apt-get install python3-dev
    sudo apt-get install libboost-python-dev
    sudo apt-get install python3-pip
    sudo apt-get install python3-tk
    sudo apt-get install libssl-dev
    sudo apt-get install kate # I don't have kate and will use it, you can use whatever you like
    sudo pip3 install matplotlib
    sudo pip3 install cmake --upgrade
    sudo pip3 install pyfhel
    sudo pip3 install pycryptodome
    sudo pip3 install scikit-learn
   After installing the necessary packages, we will prepare the libraries for the next steps. 
   

    cd palisade-development
    git checkout fab4f245f49f821a941f605f57a40c78e3fa7d60
    
   We have to checkout that commit because that commit works with the Palisade Python Wrapper library.

After that, we will go to the wrapper library

    cd ..
    cd palisade-python-demo
    git checkout 13de1a44da5f7dfe56dc2da5e821580c59576157
When we do this, we have to build and install the libraries. Before we do that, the package manager doesn't install the latest version of cmake. Because of that, we have to build and install cmake ourselves.

 1. Go to [cmake](https://www.cmake.org/download)
 2. Download
 3. Go to the download location and extract.
 4. cd into the extracted folder

Now we are supposed to build the program and install it. **NOTE**: gmake has an argument -j next to it, that says to the terminal to run building using all of the available CPU threads. If you're running this in a VM, this will most likely block your system. If you add a number (i.e. -j2) that will limit the amount of threads that are executing in one single moment. Running without -j will take a lot more time.

    ./configure
    gmake [-j]
    sudo gmake install  
   
   After installing the latest version of cmake, we are going to begin building the necessary libraries. Go to the root of the SEAL project, for me it's:
   

    cd ~/Desktop/SEAL-Python
  Install the requirements
  

    pip3 install -r requirements.txt
   Clone the submodules of the project
   

    git submodule update --init --recursive
   
   Before building the library, we must firsly enable pickling data, or rather - serialization of objects of the SEAL library. For that to happen, run the following command in the root directory. This is necessary for this project, it might not be necessary for yours.

    python helper.py
   
   Build the library
   

    cd SEAL
    cmake -S . -B build -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF -DSEAL_USE_ZSTD=OFF
    cmake --build build [-j]
   Go back to the root folder and copy-paste the output of *pwd*
   

    cd ..
    pwd
    # Copy the output like /home/user/Desktop/SEAL-Python
   We now have to change some paths in setup.py. We are doing this because it will be easier for us when we try to run our project from wherever.
   

    kate setup.py
  This will open an editor in which we have to edit lines: 12, 14, 16, 20 (if on Windows), 30. Change the path to absolute instead of relative.
  After we've done that, you can run now the following command. At this point it's not really necessary, this will come into action later for our project.
  

    python3 setup.py build_ext -i 
 At this point, the library is ready to use. We will now install the palisade library. 
 

    cd ..
    cd palisade-development

Go to the root of the library and execute the following commands. Note that the building process takes a long time. Using -j will speed up the process, but beware of the situation I mentioned before.

    git submodule update --init --recursive
    mkdir build
    cd build
    cmake ..
    make [-j]
    sudo make install
   The library is now installed and ready to use. Now we have to build the wrapper library. Go to the root and follow mostly the same steps:
   

    cd ~/Desktop/palisade-python-demo
    mkdir build
    cd build
    cmake ..
    make -j
   Instead of installing, we cd into the root
   

    cd ..
   And execute the pwd command and copy the path. We are going to change the path in *setuppython* to absoulte.
   

    pwd
    # Copy output
    kate setuppython
   Change $(pwd) to the output of pwd
   After that, the library is ready to be used, but in every new terminal instance, you are supposed to execute
   

    source setuppython
  For simplicity's sake, we copy the setuppython file from this project and the setup.py file from the SEAL project into the root of our project (where our terminal will be positioned).

After that being said, copy your versions of the aforementioned files into the clone of this project. Open the terminal and write

    source setuppython
    python3 setup.py build_ext -i
  
 Do not close the terminal after that. Now you can run any runnable file in the src/.  Every time you close the the terminal, run the `source setuppython` line. The  `python3 setup.py build_ext -i` is not necessary because that line generates a **.o** file which is used by the project implicitly.

