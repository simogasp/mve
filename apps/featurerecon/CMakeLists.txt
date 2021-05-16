add_executable(featurerecon featurerecon.cc)

target_link_libraries(featurerecon PUBLIC mve::mve_mve mve::mve_util mve::mve_sfm mve::mve_math)

install(TARGETS featurerecon DESTINATION ${CMAKE_INSTALL_BINDIR})