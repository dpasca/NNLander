# Lander04 demo
add_executable(lander05 lander05.cpp)

# Link with raylib
target_link_libraries(lander05 raylib)

# Add platform-specific options
if (MSVC)
    # Warning level 4 but don't treat as errors
    target_compile_options(lander05 PRIVATE /W4)
    # Additional optimizations
    target_compile_options(lander05 PRIVATE $<$<CONFIG:Release>:/O2 /Ob3 /Oi /Ot /GL /fp:fast /Gw /Gy>)
else()
    # Extensive warnings but don't treat as errors
    target_compile_options(lander05 PRIVATE -Wall -Wextra)
    # Additional optimizations
    target_compile_options(lander05 PRIVATE $<$<CONFIG:Release>:-O3 -march=native -flto -ffast-math -funroll-loops>)
endif()

# Installation rules
install(TARGETS lander05 DESTINATION bin) 
