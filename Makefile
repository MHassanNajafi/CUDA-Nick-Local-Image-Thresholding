# Add source files here
EXECUTABLE	:= Nick
# Cuda source files (compiled with cudacc)
CUFILES		:= Nick.cu
CUDEPS		:= Nick.cu Nick.h
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= Nick_gold.cpp 
CDEPS 		:= Nick.h
################################################################################
# Rules and targets

include ../../common/common.mk
