# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yolov5_Ascend_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yolov5_Ascend_cpp/build

# Include any dependencies generated for this target.
include src/CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/main.dir/flags.make

src/CMakeFiles/main.dir/utils.cpp.o: src/CMakeFiles/main.dir/flags.make
src/CMakeFiles/main.dir/utils.cpp.o: ../src/utils.cpp
src/CMakeFiles/main.dir/utils.cpp.o: src/CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yolov5_Ascend_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/main.dir/utils.cpp.o"
	cd /home/yolov5_Ascend_cpp/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/main.dir/utils.cpp.o -MF CMakeFiles/main.dir/utils.cpp.o.d -o CMakeFiles/main.dir/utils.cpp.o -c /home/yolov5_Ascend_cpp/src/utils.cpp

src/CMakeFiles/main.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/utils.cpp.i"
	cd /home/yolov5_Ascend_cpp/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yolov5_Ascend_cpp/src/utils.cpp > CMakeFiles/main.dir/utils.cpp.i

src/CMakeFiles/main.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/utils.cpp.s"
	cd /home/yolov5_Ascend_cpp/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yolov5_Ascend_cpp/src/utils.cpp -o CMakeFiles/main.dir/utils.cpp.s

src/CMakeFiles/main.dir/model_process.cpp.o: src/CMakeFiles/main.dir/flags.make
src/CMakeFiles/main.dir/model_process.cpp.o: ../src/model_process.cpp
src/CMakeFiles/main.dir/model_process.cpp.o: src/CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yolov5_Ascend_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/main.dir/model_process.cpp.o"
	cd /home/yolov5_Ascend_cpp/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/main.dir/model_process.cpp.o -MF CMakeFiles/main.dir/model_process.cpp.o.d -o CMakeFiles/main.dir/model_process.cpp.o -c /home/yolov5_Ascend_cpp/src/model_process.cpp

src/CMakeFiles/main.dir/model_process.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/model_process.cpp.i"
	cd /home/yolov5_Ascend_cpp/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yolov5_Ascend_cpp/src/model_process.cpp > CMakeFiles/main.dir/model_process.cpp.i

src/CMakeFiles/main.dir/model_process.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/model_process.cpp.s"
	cd /home/yolov5_Ascend_cpp/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yolov5_Ascend_cpp/src/model_process.cpp -o CMakeFiles/main.dir/model_process.cpp.s

src/CMakeFiles/main.dir/do_process.cpp.o: src/CMakeFiles/main.dir/flags.make
src/CMakeFiles/main.dir/do_process.cpp.o: ../src/do_process.cpp
src/CMakeFiles/main.dir/do_process.cpp.o: src/CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yolov5_Ascend_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/main.dir/do_process.cpp.o"
	cd /home/yolov5_Ascend_cpp/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/main.dir/do_process.cpp.o -MF CMakeFiles/main.dir/do_process.cpp.o.d -o CMakeFiles/main.dir/do_process.cpp.o -c /home/yolov5_Ascend_cpp/src/do_process.cpp

src/CMakeFiles/main.dir/do_process.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/do_process.cpp.i"
	cd /home/yolov5_Ascend_cpp/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yolov5_Ascend_cpp/src/do_process.cpp > CMakeFiles/main.dir/do_process.cpp.i

src/CMakeFiles/main.dir/do_process.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/do_process.cpp.s"
	cd /home/yolov5_Ascend_cpp/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yolov5_Ascend_cpp/src/do_process.cpp -o CMakeFiles/main.dir/do_process.cpp.s

src/CMakeFiles/main.dir/main.cpp.o: src/CMakeFiles/main.dir/flags.make
src/CMakeFiles/main.dir/main.cpp.o: ../src/main.cpp
src/CMakeFiles/main.dir/main.cpp.o: src/CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yolov5_Ascend_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/main.dir/main.cpp.o"
	cd /home/yolov5_Ascend_cpp/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/main.dir/main.cpp.o -MF CMakeFiles/main.dir/main.cpp.o.d -o CMakeFiles/main.dir/main.cpp.o -c /home/yolov5_Ascend_cpp/src/main.cpp

src/CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	cd /home/yolov5_Ascend_cpp/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yolov5_Ascend_cpp/src/main.cpp > CMakeFiles/main.dir/main.cpp.i

src/CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	cd /home/yolov5_Ascend_cpp/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yolov5_Ascend_cpp/src/main.cpp -o CMakeFiles/main.dir/main.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/utils.cpp.o" \
"CMakeFiles/main.dir/model_process.cpp.o" \
"CMakeFiles/main.dir/do_process.cpp.o" \
"CMakeFiles/main.dir/main.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

../out/main: src/CMakeFiles/main.dir/utils.cpp.o
../out/main: src/CMakeFiles/main.dir/model_process.cpp.o
../out/main: src/CMakeFiles/main.dir/do_process.cpp.o
../out/main: src/CMakeFiles/main.dir/main.cpp.o
../out/main: src/CMakeFiles/main.dir/build.make
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_alphamat.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_aruco.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_barcode.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_bgsegm.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_bioinspired.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_ccalib.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_dnn_objdetect.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_dnn_superres.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_dpm.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_face.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_freetype.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_fuzzy.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_hdf.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_hfs.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_img_hash.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_intensity_transform.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_line_descriptor.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_mcc.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_quality.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_rapid.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_reg.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_rgbd.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_saliency.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_shape.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_stereo.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_structured_light.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_superres.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_surface_matching.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_tracking.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_videostab.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_viz.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_wechat_qrcode.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_xobjdetect.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_xphoto.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_datasets.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_plot.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_text.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_phase_unwrapping.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_optflow.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_ximgproc.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5.4d
../out/main: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.5.4d
../out/main: /usr/lib/gcc/aarch64-linux-gnu/11/libgomp.so
../out/main: /usr/lib/aarch64-linux-gnu/libpthread.a
../out/main: src/CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yolov5_Ascend_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable ../../out/main"
	cd /home/yolov5_Ascend_cpp/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/main.dir/build: ../out/main
.PHONY : src/CMakeFiles/main.dir/build

src/CMakeFiles/main.dir/clean:
	cd /home/yolov5_Ascend_cpp/build/src && $(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/main.dir/clean

src/CMakeFiles/main.dir/depend:
	cd /home/yolov5_Ascend_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yolov5_Ascend_cpp /home/yolov5_Ascend_cpp/src /home/yolov5_Ascend_cpp/build /home/yolov5_Ascend_cpp/build/src /home/yolov5_Ascend_cpp/build/src/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/main.dir/depend

