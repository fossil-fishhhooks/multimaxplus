#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;

Field evalEffectiveField(const Ferromagnet*);
Field evalEffectiveFieldLocalSlab(const Ferromagnet*);
FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet*);
