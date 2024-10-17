#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <barrier>
#include <thread>

#include <thread>
#include <vector>
#include <barrier>

#include "grid.h"
#include "input.h"
#include "input_set.h"

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::steady_clock;
using std::this_thread::sleep_for;
using TimePoint = std::chrono::steady_clock::time_point;
using TimeSpan = std::chrono::duration<double>;

void update_energy_at(
    Grid &prev_state,
    Grid &next_state,
    Input const &input,
    size_t x,
    size_t y,
    size_t t
);

Grid parallel_simulate(Input const &input, unsigned int display, size_t thread_count);
Grid serial_simulate(Input const &input, unsigned int display);
void compare_grids(Grid &serial_grid, Grid &parallel_grid);



// desc: Runs the same problem serially and in parallel, then compares their outputs.
// pre : None.
// post: None, aside from description.
int main(int argc, char *argv[]) {

    if (argc != 2) {
        std::cout << "Program requires exactly one input between 0 and 3." << std::endl;
        return 1;
    }

    size_t selection = atoi(argv[1]);

    if (selection > 3) {
        std::cout << "Program requires exactly one input between 0 and 3." << std::endl;
        return 1;
    }

    std::vector<size_t> durations = {64, 128, 256, 512, 1024, 2048, 4096};
    std::vector<size_t> thread_counts = {1, 2, 3, 4, 5, 6, 7, 8};

    std::cout << "Duration, Thread Count, Runtime (ms)" << std::endl;

    Input *example_inputs[] = {
        new TopLeft(16,16,64),
        new Orbit(64,4096),
        new Wall(32,8192),
        // new OrbitWall(128,4096)
        // new OrbitWall(128,64)
        new OrbitWall(128,64)
    };

    for (size_t duration : durations) {
        for (size_t thread_count : thread_counts) {
            Input *example_inputs[] = {
                new TopLeft(16,16,64),
                new Orbit(64,4096),
                new Wall(32,8192),
                // new OrbitWall(128,4096)
                new OrbitWall(128,duration)
            };
            // WallOrbit input(128, duration);
            Input & input = *(example_inputs[selection]);

            TimePoint start_time_parallel_simulate = steady_clock::now();
            Grid result = parallel_simulate(input, 0, thread_count);
            TimePoint end_time_parallel_simulate = steady_clock::now();
            TimeSpan runtime = duration_cast<TimeSpan>(end_time_parallel_simulate - start_time_parallel_simulate);

            std::cout << duration << ", " << thread_count << ", " << runtime.count() << std::endl;
        }
    }

    // Input & input = *(example_inputs[selection]);

    // std::cout << "Running serial simulation." << std::endl;
    // TimePoint start_time_serial_simulate = steady_clock::now();
    // Grid serial_result   = serial_simulate(input,input.get_duration()/64);
    // TimePoint end_time_serial_simulate = steady_clock::now();
    // TimeSpan span_serial_simulate = duration_cast<TimeSpan>(end_time_serial_simulate - start_time_serial_simulate);

    // std::cout << "Running parallel simulation." << std::endl;
    // TimePoint start_time_parallel_simulate = steady_clock::now();
    // // Grid parallel_result = parallel_simulate(input,input.get_duration()/64,2);
    // Grid parallel_result = parallel_simulate(input,0,2);
    // TimePoint end_time_parallel_simulate = steady_clock::now();
    // TimeSpan span_parallel_simulate = duration_cast<TimeSpan>(end_time_parallel_simulate - start_time_parallel_simulate);

    // std::cout << "Comparing grids." << std::endl;
    // compare_grids(serial_result,parallel_result);

    // std::cout << "Serial Execution time is : " << span_serial_simulate.count() << '\n';
    // std::cout << "Parallel Execution time is : " << span_parallel_simulate.count() << '\n';
    return 0;
}

// Simulate the heat transfer for a range of rows.
void simulate(size_t start, size_t end, size_t width, size_t duration, Grid &a, Grid &b, Input const &input, std::barrier<> &sync_point)
{
    // Simulate the heat transfer for the specified range of rows.
    for (size_t t = 0; t < duration; ++t) {

        // Determine which state is the current state and which is the next state.
        Grid &prev_state = ((t % 2) == 0) ? a : b;
        Grid &next_state = ((t % 2) == 0) ? b : a;

        // Simulate the heat transfer for the specified range of rows.
        for (size_t y = start; y < end; ++y) {

            // Simulate the heat transfer for each cell in the row.
            for (size_t x = 0; x < width; ++x) {

                // Update the energy at the current cell.
                next_state.at(x, y) = 0;
                update_energy_at(prev_state, next_state, input, x, y, t);
            }
        }

        // Wait for all threads to finish before moving to the next time step.
        sync_point.arrive_and_wait();
    }
}

// Display the simulation state at regular intervals.
void display_simulation(size_t duration, unsigned int display, Grid &a, Grid &b, std::barrier<> &sync_point)
{
    for (size_t t = 0; t < duration; ++t) {
        if ((t % display) == 0) {

            // Display the current state of the simulation.
            Grid &prev_state = ((t % 2) == 0) ? a : b;
            prev_state.display();
        }

        // Wait for all threads to finish before moving to the next time step.
        sync_point.arrive_and_wait();
    }

    // Display the final state of the simulation.
    Grid &last_state = ((duration % 2) == 0) ? a : b;
    last_state.display();
}

// This is the function that you need to refactor for parallelism!
//
// Currently, its definition matches the `serial_simulate` function.
// You may create additional functions for this function to call into, and
// you may feel free to modify the content of this function.
//
// HOWEVER:
//  - The signature of this function should not be modified!
//  - Calling `parallel_simulate` should be all that is needed to perform simulation
//    with the provided inputs.
//  - The output of `parallel_simulate` should match the output of `serial_simulate`.
Grid parallel_simulate(Input const &input, unsigned int display, size_t thread_count)
{
    size_t width    = input.get_width();
    size_t height   = input.get_height();
    size_t duration = input.get_duration();

    Grid a(width,height);
    Grid b(width,height);

    a.clear();

    // A barrier to synchronize threads at the end of each time step.
    std::barrier sync_point(thread_count + (display > 0 ? 1 : 0));

    // Create threads to simulate the heat transfer for different ranges of rows.
    std::vector<std::thread> threads;
    size_t rows_per_thread = height / thread_count;
    
    for (size_t i = 0; i < thread_count; ++i) {

        // Calculate the range of rows to simulate for this thread.
        size_t start = i * rows_per_thread;
        size_t end = (i == thread_count - 1) ? height : start + rows_per_thread;
        threads.emplace_back(simulate, start, end, width, duration, std::ref(a), std::ref(b), std::cref(input), std::ref(sync_point));
    }

    // Create a thread to display the simulation state at regular intervals.
    if (display > 0) {
        threads.emplace_back(display_simulation, duration, display, std::ref(a), std::ref(b), std::ref(sync_point));
    }

    // Wait for all threads to finish.
    for (auto &thread : threads) {
        thread.join();
    }

    // Return the final state of the simulation.
    Grid &last_state = ((duration % 2) == 0) ? a : b;
    return last_state;
}




// desc: Set the element at (x,y) in `next_state` to match the state
//       that should follow for element (x,y) in `prev_state`.
// pre : `prev_state` and `next_state` must have the same dimensions,
//       and (x,y) must be valid coordinates within both. Also, the
//       (x,y,t) coordinates provided must be valid in `input`.
// post: none, aside from description
void update_energy_at(
    Grid &prev_state,
    Grid &next_state,
    Input const &input,
    size_t x,
    size_t y,
    size_t t
) {
    double const rval = input.conductivity_at(x,y,t);
    double const lost = rval * 0.25;
    double const kept = 1.0 - rval;
    // For our purposes, energy doesn't dissipate across diagonals
    double const weights[] = {kept,lost,0};

    // Iterate across the (3x3) cell neighborhood around (x,y)
    for (int i=-1; i<=1; i++) {
        for (int j=-1; j<=1; j++) {
            // Determine weight for energy transfer between
            // the current pair of cells
            size_t weight_index = abs(i) + abs(j);
            double w = weights[weight_index];

            size_t cell_y = y + i;
            size_t cell_x = x + j;
            if (next_state.contains(cell_x,cell_y)) {
                // Accumulate energy transferred from adjacent cells
                next_state.at(x,y) += w * prev_state.at(cell_x,cell_y);
            }
        }
    }
    // After handling energy dissipation, add in the energy for this
    // time step.
    next_state.at(x,y) += input.energy_at(x,y,t);
}


// desc: Performs a serial heat transfer simulation, as described by the input
//       `input`. If `display` is positive, then the state of the simulation
//        is displayed once every `display` iterations, as well as at the
//        final iteration.
// pre : None.
// post: None, aside from description.
Grid serial_simulate(Input const &input, unsigned int display)
{
    size_t width    = input.get_width();
    size_t height   = input.get_height();
    size_t duration = input.get_duration();

    Grid a(width,height);
    Grid b(width,height);

    a.clear();

    for (size_t t=0; t<duration; t++) {
        Grid &prev_state = ((t%2)==0) ? a : b;
        Grid &next_state = ((t%2)==0) ? b : a;

        // If display is enabled, simulation should be shown before every
        // time step
        if ( (display > 0 ) && ((t%display)==0) ) {
            prev_state.display();
        }

        for (size_t y=0; y<height; y++) {
            for (size_t x=0; x<width; x++) {
                next_state.at(x,y) = 0;
                update_energy_at(prev_state,next_state,input,x,y,t);
            }
        }
    }

    Grid &last_state = ((duration%2)==0) ? a : b;

    // Display the final state of the simulation if display is enabled.
    if (display > 0) {
        last_state.display();
    }

    return last_state;
}


// desc: Creates a grid representing the element-wise absolute difference
//       between grids, displays it, then reports the total and average
//       difference between elements.
// pre : None.
// post: None, aside from description.
void compare_grids(Grid &serial_grid, Grid &parallel_grid)
{
    size_t width  = serial_grid.get_width();
    size_t height = serial_grid.get_height();
    Grid delta(width,height);

    double total_difference = 0;
    for (size_t y=0; y<height; y++) {
        for (size_t x=0; x<width; x++) {
            double diff = fabsl(serial_grid.at(x,y)-parallel_grid.at(x,y));
            total_difference += diff;
            delta.at(x,y) = diff * 10;
        }
    }

    delta.display();
    std::cout << "Total difference: " << total_difference << std::endl;
    std::cout << "Average difference per value: "
              << total_difference / (height*width)
              << std::endl;
}


