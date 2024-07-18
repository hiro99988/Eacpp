#ifndef _HAS_RNG_H
#define _HAS_RNG_H

#include "Rng/IRng.h"
#include "Rng/Rng.h"

namespace Eacpp {

class HasRng {
   protected:
    IRng* _rng;

    HasRng() {
        _rng = new Rng();
        _isRngCreated = true;
    }
    HasRng(IRng* rng) : _rng(rng) {}
    virtual ~HasRng() {
        if (_isRngCreated) {
            delete _rng;
        }
    }

   private:
    bool _isRngCreated = false;
};
}  // namespace Eacpp

#endif