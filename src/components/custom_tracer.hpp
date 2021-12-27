#pragma once

#include <iostream>

class custom_tracer {

    public:
    virtual void to_stream_header(std::string name, std::ostream& os) const  = 0;
    virtual void to_stream(std::ostream& os) const  = 0;
    virtual void from_stream(std::istream& is) = 0;

    virtual ~custom_tracer() {}

};


