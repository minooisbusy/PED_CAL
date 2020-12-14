add_executable(run_ped main.cpp)
target_link_libraries(run_ped rectangle)
target_link_libraries(run_ped ${OpenCV_LIBS})