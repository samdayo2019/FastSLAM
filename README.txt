Student: Samuel Dayo 
McGill ID: 260889723

Modify the following make file parameters to point to the location of your installed system C library: 
IFLAGS: modify to point to the include folder of your installed System C library
LFLAGS: modify to point to the library folder of your installed System C library 

To compile the program simply execute: make. The following commands will be executed based on the Makefile: 
g++ -c -o system.o system.cpp -g3 -isystem /usr/local/systemc232/include
g++ -o sad system.o -g3 -lsystemc -lm -L/usr/local/systemc232/lib-linux64

To execute the program and to include additional timing, run the following command:
time ./sad mem_init.txt [[[addrC addrA addrB] size] loops]

Here addrC, addrA and addrC are optional parameters that can be included to override the default starting addresses and output address used in the matrix multiplication computation. Size is the size of the rows and the arrays, and loops overrides the number of loopsd to be completed. 

The time command will print out the real, user, and system time taken to run the program. 

Note: if the entered command is invalid, you will be prompted to enter a valid input.



