#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"
#include "HepMC3/WriterAscii.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  if (argc != 7) {
    std::cerr
        << "Usage: " << argv[0]
        << " OUTPUT_HEPMC OUTPUT_JSON N_EVENTS SEED"
        << " PTHAT_MIN PTHAT_MAX\n";
    return 1;
  }

  const std::string outputHepMC = argv[1];
  const std::string outputJson = argv[2];
  const int nEvents = std::stoi(argv[3]);
  const int seed = std::stoi(argv[4]);
  const double pTHatMin = std::stod(argv[5]);
  const double pTHatMax = std::stod(argv[6]);

  if (
      nEvents <= 0 ||
      seed <= 0 ||
      pTHatMin < 0.0 ||
      (pTHatMax > 0.0 && pTHatMax <= pTHatMin)
  ) {
    std::cerr << "ERROR: invalid numerical argument\n";
    return 2;
  }

  Pythia8::Pythia pythia;

  auto requireSetting = [&pythia](const std::string& setting) {
    if (!pythia.readString(setting)) {
      std::cerr
          << "ERROR: Pythia rejected setting: "
          << setting << "\n";
      std::exit(3);
    }
  };

  requireSetting("Beams:idA = 2212");
  requireSetting("Beams:idB = 2212");
  requireSetting("Beams:eCM = 13000.");
  requireSetting("HardQCD:all = on");

  requireSetting(
      "PhaseSpace:pTHatMin = " +
      std::to_string(pTHatMin)
  );

  if (pTHatMax > 0.0) {
    requireSetting(
        "PhaseSpace:pTHatMax = " +
        std::to_string(pTHatMax)
    );
  }

  // Monash 2013 tune. This is not CMS CP5.
  requireSetting("Tune:ee = 7");
  requireSetting("Tune:pp = 14");

  requireSetting("Random:setSeed = on");
  requireSetting(
      "Random:seed = " +
      std::to_string(seed)
  );

  requireSetting("Next:numberShowInfo = 0");
  requireSetting("Next:numberShowProcess = 0");
  requireSetting("Next:numberShowEvent = 0");
  requireSetting("Main:timesAllowErrors = 100000");

  if (!pythia.init()) {
    std::cerr
        << "ERROR: Pythia initialization failed\n";
    return 4;
  }

  HepMC3::Pythia8ToHepMC3 converter;
  converter.set_print_inconsistency(false);
  converter.set_free_parton_warnings(false);
  converter.set_store_xsec(false);

  HepMC3::WriterAscii writer(outputHepMC);

  int nWritten = 0;
  int nAttempts = 0;
  int nFailures = 0;
  const int maxFailures = 1000 + 20 * nEvents;

  while (nWritten < nEvents) {
    ++nAttempts;

    if (!pythia.next()) {
      ++nFailures;

      if (nFailures > maxFailures) {
        std::cerr
            << "ERROR: too many failed attempts\n";
        return 5;
      }

      continue;
    }

    HepMC3::GenEvent event(
        HepMC3::Units::GEV,
        HepMC3::Units::MM
    );

    converter.fill_next_event(pythia, &event);
    writer.write_event(event);
    ++nWritten;
  }

  writer.close();
  pythia.stat();

  // Pythia reports sigmaGen in millibarns.
  const double sigmaMb = pythia.info.sigmaGen();
  const double sigmaErrMb = pythia.info.sigmaErr();
  const double sigmaPb = sigmaMb * 1.0e9;
  const double sigmaErrPb = sigmaErrMb * 1.0e9;

  std::ofstream metadata(outputJson);

  if (!metadata) {
    std::cerr
        << "ERROR: cannot write metadata JSON\n";
    return 6;
  }

  metadata << std::setprecision(17);
  metadata << "{\n";
  metadata << "  \"generator\": \"Pythia8\",\n";
  metadata << "  \"process\": \"HardQCD:all\",\n";
  metadata << "  \"sqrt_s_GeV\": 13000.0,\n";
  metadata << "  \"tune\": "
           << "\"Monash2013_TuneEE7_TunePP14\",\n";
  metadata << "  \"n_events\": "
           << nWritten << ",\n";
  metadata << "  \"n_attempts\": "
           << nAttempts << ",\n";
  metadata << "  \"n_failed_attempts\": "
           << nFailures << ",\n";
  metadata << "  \"seed\": "
           << seed << ",\n";
  metadata << "  \"pthat_min_GeV\": "
           << pTHatMin << ",\n";

  if (pTHatMax > 0.0) {
    metadata << "  \"pthat_max_GeV\": "
             << pTHatMax << ",\n";
  } else {
    metadata << "  \"pthat_max_GeV\": null,\n";
  }

  metadata << "  \"sigma_gen_mb\": "
           << sigmaMb << ",\n";
  metadata << "  \"sigma_err_mb\": "
           << sigmaErrMb << ",\n";
  metadata << "  \"sigma_gen_pb\": "
           << sigmaPb << ",\n";
  metadata << "  \"sigma_err_pb\": "
           << sigmaErrPb << ",\n";
  metadata << "  \"physics_role\": "
           << "\"inclusive_QCD_importance_stratum\",\n";
  metadata << "  \"generator_filter\": \"none\"\n";
  metadata << "}\n";

  std::cout
      << "WROTE_HEPMC=" << outputHepMC << "\n"
      << "WROTE_METADATA=" << outputJson << "\n"
      << "EVENTS=" << nWritten << "\n"
      << "SIGMA_PB=" << sigmaPb << "\n";

  return 0;
}
