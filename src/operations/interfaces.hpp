#pragma once

class ProxyMPI {
  public:
    virtual void acquire() {}
    virtual void release() {}
    virtual ~ProxyMPI() = default;
};