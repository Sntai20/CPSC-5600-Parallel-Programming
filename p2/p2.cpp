#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <barrier>
#include <thread>


#include "grid.h"
#include "input.h"
#include "input_set.h"



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

    Input *example_inputs[] = {
        new TopLeft(16,16,64),
        new Orbit(64,4096),
        new Wall(32,8192),
        new OrbitWall(128,4096)
    };

    Input & input = *(example_inputs[selection]);


    std::cout << "Running serial simulation." << std::endl;
    Grid serial_result   = serial_simulate(input,input.get_duration()/64);
    std::cout << "Running parallel simulation." << std::endl;
    Grid parallel_result = parallel_simulate(input,input.get_duration()/64,2);
    std::cout << "Comparing grids." << std::endl;
    compare_grids(serial_result,parallel_result);

    return 0;
}


// This is the function that you need to refactor for parallelism!
//
// Currenty, its definition matches the `serial_simulate` function.
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




// desc: Set the element at (x,y) in `next_state` to match the state
//       that should follow for element (x,y) in `prev_state`.
// pre : `prev_state` and `next_state` must have the same dimensions,
//       and (x,y) must be valid coodinates within both. Also, the
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
            // Determine weight for energy tranfer between
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


