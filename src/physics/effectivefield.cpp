#include "effectivefield.hpp"

#include "afmexchange.hpp"
#include "anisotropy.hpp"
#include "antiferromagnet.hpp"
#include "atmexchange.hpp"
#include "demag.hpp"
#include "dmi.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "local_dmi.hpp"
#include "magnetoelasticfield.hpp"
#include "zeeman.hpp"
#include "stepprofile.hpp"

Field evalEffectiveField(const Ferromagnet* magnet) {
  StepCheckpoint checkpoint;
  auto& timings = stepTimings();
  // there will probably be exchange, otherwise safely initialized as 0
  Field h = evalExchangeField(magnet);
  timings.exchange += checkpoint();
  if (!anisotropyAssuredZero(magnet)) {
    checkpoint();
    Field contribution = evalAnisotropyField(magnet);
    timings.otherFields += checkpoint();
    h += contribution;
    timings.fieldAdd += checkpoint();
  }
  if (!externalFieldAssuredZero(magnet)) {
    checkpoint();
    Field contribution = evalExternalField(magnet);
    timings.external += checkpoint();
    h += contribution;
    timings.fieldAdd += checkpoint();
  }
  if (!inhomoDmiAssuredZero(magnet)) {
    checkpoint();
    Field contribution = evalDmiField(magnet);
    timings.otherFields += checkpoint();
    h += contribution;
    timings.fieldAdd += checkpoint();
  }
  if (!demagFieldAssuredZero(magnet)) {
    checkpoint();
    Field contribution = evalDemagField(magnet);
    timings.demag += checkpoint();
    h += contribution;
    timings.fieldAdd += checkpoint();
  }
  if (!magnetoelasticAssuredZero(magnet)) {
    checkpoint();
    Field contribution = evalMagnetoelasticField(magnet);
    timings.otherFields += checkpoint();
    h += contribution;
    timings.fieldAdd += checkpoint();
  }
  if (magnet->isSublattice()) {
    // AFM exchange terms
    if (!inHomoAfmExchangeAssuredZero(magnet)) {
      checkpoint();
      Field contribution = evalInHomogeneousAfmExchangeField(magnet);
      timings.otherFields += checkpoint();
      h += contribution;
      timings.fieldAdd += checkpoint();
    }
  }
  if (!homoAfmExchangeAssuredZero(magnet)) {
    checkpoint();
    Field contribution = evalHomogeneousAfmExchangeField(magnet);
    timings.otherFields += checkpoint();
    h += contribution;
    timings.fieldAdd += checkpoint();
  }
  // Homogeneous (local) DMI term
  if (!homoDmiAssuredZero(magnet)) {
    checkpoint();
    Field contribution = evalHomoDmiField(magnet);
    timings.otherFields += checkpoint();
    h += contribution;
    timings.fieldAdd += checkpoint();
  }
  // ATM anisotropic exchange
  if (!atmExchangeAssuredZero(magnet)) {
    checkpoint();
    Field contribution = evalAtmExchangeField(magnet);
    timings.otherFields += checkpoint();
    h += contribution;
    timings.fieldAdd += checkpoint();
  }
  return h;
}

FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalEffectiveField, 3, "effective_field", "T");
}
