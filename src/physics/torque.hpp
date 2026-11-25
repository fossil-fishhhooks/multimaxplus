#pragma once

#include "quantityevaluator.hpp"
#include "reduce.hpp"

class Ferromagnet;
class Field;
class DynamicEquation;

// Deterministic ordinary-FM equations; stage magnetization remains replicated.
const Ferromagnet* localRkMagnet(const DynamicEquation&);
Field evalLocalTorque(const DynamicEquation&);

Field evalTorque(const Ferromagnet*);
Field evalLlgTorque(const Ferromagnet*);
Field evalRelaxTorque(const Ferromagnet*);
real evalMaxTorque(const Ferromagnet*);

FM_FieldQuantity torqueQuantity(const Ferromagnet*);
FM_FieldQuantity llgTorqueQuantity(const Ferromagnet*);
FM_FieldQuantity relaxTorqueQuantity(const Ferromagnet*);
FM_ScalarQuantity maxTorqueQuantity(const Ferromagnet*);
