cmake_minimum_required(VERSION 3.12)
project(lwpr)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_STATIC_LIBS "Build static libraries" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_PYTHON "Build Python bindings" ON)
option(USE_EXPAT "Use libexpat for XML support" ON)

set(NUM_THREADS 1 CACHE STRING "Number of execution threads")

# RPATH settings for installed libraries
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

set(LWPR_AUTHOR sethu.vijayakumar@ed.ac.uk)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Threads REQUIRED)

if(USE_EXPAT)
    find_package(EXPAT)
    set(HAVE_LIBEXPAT ${EXPAT_FOUND})
endif()


add_definitions(-DHAVE_CONFIG_H)

include_directories(
    include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${EXPAT_INCLUDE_DIRS}
)

set(LWPR_VERSION_STRING 1.2.6)
set(LWPR_SOVERSION_STRING 1)

if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/include/lwpr_config.h)
    message(FATAL_ERROR "'include/lwpr_config.h' exists in the source path. Delete it before building.")
endif()

configure_file(include/lwpr_config.h.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/lwpr_config.h)

set(LWPR_SOURCES
    src/lwpr.c
    src/lwpr_aux.c
    src/lwpr_binio.c
    src/lwpr_math.c
    src/lwpr_mem.c
    src/lwpr_xml.c
)

# Shared library
if(BUILD_SHARED_LIBS)
    add_library(lwpr SHARED ${LWPR_SOURCES})
    set_target_properties(lwpr PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

    target_link_libraries(lwpr ${CMAKE_THREAD_LIBS_INIT} m ${EXPAT_LIBRARIES})

    set_target_properties(lwpr PROPERTIES 
        VERSION ${LWPR_VERSION_STRING} 
        SOVERSION ${LWPR_SOVERSION_STRING}
    )
endif()

# Static library
if(BUILD_STATIC_LIBS)
    add_library(lwpr_static STATIC ${LWPR_SOURCES})
    if(MSVC)
        set_target_properties(lwpr_static PROPERTIES OUTPUT_NAME "lwpr_static")
    else()
        set_target_properties(lwpr_static PROPERTIES OUTPUT_NAME "lwpr")
    endif()
    set_target_properties(lwpr_static PROPERTIES VERSION ${LWPR_VERSION_STRING})
endif()

# Examples
if(BUILD_EXAMPLES)
    add_executable(example_cpp example_cpp/cross.cc)
    target_link_libraries(example_cpp lwpr)
endif()

# Python bindings
if(BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
    include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

    add_library(pylwpr MODULE python/lwprmodule.c)
    target_link_libraries(pylwpr lwpr ${Python3_LIBRARIES})

    set_target_properties(pylwpr PROPERTIES
        PREFIX ""
        OUTPUT_NAME "pylwpr"
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
    )

    add_dependencies(pylwpr lwpr)

    # Overridable with -DPYTHON_SITE_PACKAGES=<dir>. Without this the module
    # always installs into the ACTIVE interpreter's site-packages, so a build
    # into a throwaway prefix still writes to the user's environment -- and
    # leaves it with an RPATH pointing at a directory that is about to be
    # deleted. The verifier depends on being able to redirect this.
    if(NOT DEFINED PYTHON_SITE_PACKAGES)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
            OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()
    install(TARGETS pylwpr LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
    install(TARGETS lwpr RUNTIME DESTINATION ${PYTHON_SITE_PACKAGES})
endif()

# Install rules
if(BUILD_SHARED_LIBS)
    install(TARGETS lwpr DESTINATION lib)
endif()

if(BUILD_STATIC_LIBS)
    install(TARGETS lwpr_static DESTINATION lib)
endif()

if(BUILD_EXAMPLES)
    install(TARGETS example_cpp RUNTIME DESTINATION bin)
endif()
