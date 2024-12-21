add_executable(test_ops test_ops.cpp)

target_include_directories(test_ops PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(test_ops torch_nupu torch_cpu)
# TODO: deal rpath to `torch_cpu`
set_target_properties(test_ops PROPERTIES INSTALL_RPATH "${RPATH_ORIGIN}/../lib")

install(TARGETS test_ops RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
