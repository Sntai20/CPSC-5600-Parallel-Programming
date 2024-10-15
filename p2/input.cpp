// Copyright (c) 2024 Braxton Cuneo, under MIT License
#include "input.h"

// desc: Standard constructor that initializes width, height, duration.
// pre : None.
// post: None, aside from description.
Input::Input(size_t width, size_t height, size_t duration)
    : width(width)
    , height(height)
    , duration(duration)
{}

// desc: Returns the width of the Input.
// pre : None.
// post: None, aside from description.
size_t Input::get_width() const
{
    return width;
}

// desc: Returns the height of the Input.
// pre : None.
// post: None, aside from description.
size_t Input::get_height() const
{
    return height;
}

// desc: Returns the duration of the Input.
// pre : None.
// post: None, aside from description.
size_t Input::get_duration() const
{
    return duration;
}

