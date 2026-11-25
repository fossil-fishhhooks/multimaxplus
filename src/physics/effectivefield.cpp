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
#include "strayfield.hpp"
#include "physicalslab.hpp"

static Field effectiveField(const Ferromagnet* magnet, bool localOnly) {
  StepCheckpoint checkpoint;
  auto& timings = stepTimings();
  // there will probably be exchange, otherwise safely initialized as 0
  Field h = localOnly ? evalExchangeFieldLocalSlab(magnet) : evalExchangeField(magnet);
  const auto size = magnet->grid().size();
  const auto& session = magnet->world()->session();
  const PhysicalSlab slab(size.x, size.y, size.z, session.rank(), session.size());
  auto addContribution = [&](const Field& contribution) {
    if (localOnly) addToSlab(h, 1, contribution, slab.firstCell, slab.localCells);
    else h += contribution;
  };
  timings.exchange += checkpoint();
  if (!anisotropyAssuredZero(magnet)) {
    checkpoint();
    Field contribution = evalAnisotropyField(magnet);
    timings.otherFields += checkpoint();
    addContribution(contribution);
    timings.fieldAdd += checkpoint();
  }
  if (!externalFieldAssuredZero(magnet)) {
    checkpoint();
    Field contribution = evalExternalField(magnet);
    timings.external += checkpoint();
    addContribution(contribution);
    timings.fieldAdd += checkpoint();
  }
  if (!inhomoDmiAssuredZero(magnet)) {
    checkpoint();
    Field contribution = evalDmiField(magnet);
    timings.otherFields += checkpoint();
    addContribution(contribution);
    timings.fieldAdd += checkpoint();
  }
  if (!demagFieldAssuredZero(magnet)) {
    checkpoint();
    Field contribution = localOnly ? magnet->getStrayField(magnet)->evalLocalSlab()
                                   : evalDemagField(magnet);
    timings.demag += checkpoint();
    addContribution(contribution);
    timings.fieldAdd += checkpoint();
  }
  if (!magnetoelasticAssuredZero(magnet)) {
    checkpoint();
    Field contribution = evalMagnetoelasticField(magnet);
    timings.otherFields += checkpoint();
    addContribution(contribution);
    timings.fieldAdd += checkpoint();
  }
  if (magnet->isSublattice()) {
    // AFM exchange terms
    if (!inHomoAfmExchangeAssuredZero(magnet)) {
      checkpoint();
      Field contribution = evalInHomogeneousAfmExchangeField(magnet);
      timings.otherFields += checkpoint();
      addContribution(contribution);
      timings.fieldAdd += checkpoint();
    }
  }
  if (!homoAfmExchangeAssuredZero(magnet)) {
    checkpoint();
    Field contribution = evalHomogeneousAfmExchangeField(magnet);
    timings.otherFields += checkpoint();
    addContribution(contribution);
    timings.fieldAdd += checkpoint();
  }
  // Homogeneous (local) DMI term
  if (!homoDmiAssuredZero(magnet)) {
    checkpoint();
    Field contribution = evalHomoDmiField(magnet);
    timings.otherFields += checkpoint();
    addContribution(contribution);
    timings.fieldAdd += checkpoint();
  }
  // ATM anisotropic exchange
  if (!atmExchangeAssuredZero(magnet)) {
    checkpoint();
    Field contribution = evalAtmExchangeField(magnet);
    timings.otherFields += checkpoint();
    addContribution(contribution);
    timings.fieldAdd += checkpoint();
  }
  return h;
}

Field evalEffectiveField(const Ferromagnet* magnet) { return effectiveField(magnet, false); }
Field evalEffectiveFieldLocalSlab(const Ferromagnet* magnet) {
  return effectiveField(magnet, true);
}

FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalEffectiveField, 3, "effective_field", "T");
}
