#pragma once

#include <iostream>

class custom_tracer {

    public:
    virtual void to_stream_header(std::string name, std::ostream& os) const  {}
    virtual void to_stream(std::ostream& os) const  {}
    virtual void from_stream(std::istream& is) {}

    virtual ~custom_tracer() {}

};


