Makefile Configuration of MShadow
=====
This folder contains Makefile configuration of mshadow. MShadow is a template library,
you only need to include mshadow to use it.

You can compile mshadow with different mode, for example, with or without CUDA. There are different compile flags
that you might need to set in your own configuration, and this folder provides a Makefile script to help you do that.

Usage
=====
* Set the configurations via variables in your Makefile, see example in [../guide/config.mk](../guide/config.mk)
* include [mshadow.mk](mshadow.mk) in your Makefile
* mshadow.mk will give you compiler variables that you can include when compiling
  - Add MSHADOW_CFLAGS to the compile flags
  - Add MSHADOW_LDFLAGS to the linker flags
  - Add MSHADOW_NVCCFLAGS to the nvcc compile flags
* For example Makefile, see [../guide/Makefile](../guide/Makefile)
