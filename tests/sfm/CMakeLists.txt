set(MVE_SFM_TEST_SOURCES
        ba_cholesky
        ba_conjugate_gradient
        ba_dense_vector
        ba_sparse_matrix
        bundler_tracks
        homography
        matching
        nearest_neighbor
        pose
        pose_p3p
        surf
        )

foreach(testname IN LISTS MVE_SFM_TEST_SOURCES)
    set(test_target smvs_test_sfm_${testname})
    add_executable(${test_target} gtest_${testname}.cc)
    target_link_libraries(${test_target} GTest::GTest GTest::Main mve_sfm)

    add_test(NAME ${test_target} COMMAND ${test_target})
    unset(test_target)
endforeach()