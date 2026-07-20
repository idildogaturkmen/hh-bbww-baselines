#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/WriterAscii.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  if (argc < 5 || argc > 6) {
    std::cerr
        << "Usage: " << argv[0]
        << " input.lhe output.hepmc nEvents pythiaSeed [--force-hbb]\n";
    return 1;
  }

  const std::string lhe_file = argv[1];
  const std::string hepmc_file = argv[2];
  const int n_events = std::atoi(argv[3]);
  const int pythia_seed = std::atoi(argv[4]);

  const bool force_hbb =
      argc == 6 && std::string(argv[5]) == "--force-hbb";

  if (n_events <= 0) {
    std::cerr << "ERROR: nEvents must be positive\n";
    return 2;
  }

  if (pythia_seed <= 0 || pythia_seed > 900000000) {
    std::cerr
        << "ERROR: Pythia seed must be between 1 and 900000000\n";
    return 3;
  }

  Pythia8::Pythia pythia;

  pythia.readString("Beams:frameType = 4");
  pythia.readString("Beams:LHEF = " + lhe_file);

  pythia.readString("Random:setSeed = on");
  pythia.readString(
      "Random:seed = " + std::to_string(pythia_seed)
  );

  if (force_hbb) {
    pythia.readString("25:onMode = off");
    pythia.readString("25:onIfMatch = 5 -5");
  }

  pythia.readString("PartonLevel:all = on");
  pythia.readString("PartonLevel:ISR = on");
  pythia.readString("PartonLevel:FSR = on");
  pythia.readString("HadronLevel:all = on");

  pythia.readString("Check:event = off");

  pythia.readString("Next:numberCount = 1000");
  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");

  if (!pythia.init()) {
    std::cerr << "ERROR: Pythia initialization failed\n";
    return 4;
  }

  HepMC3::WriterAscii writer(hepmc_file);
  Pythia8::Pythia8ToHepMC converter;

  int n_written = 0;

  for (int index = 0; index < n_events; ++index) {
    if (!pythia.next()) {
      std::cerr
          << "WARNING: pythia.next() failed at input event "
          << index << "\n";
      continue;
    }

    HepMC3::GenEvent event;

    converter.fill_next_event(
        pythia,
        &event
    );

    writer.write_event(event);

    ++n_written;
  }

  pythia.stat();
  writer.close();

  std::cout
      << "Requested events: " << n_events << "\n"
      << "Written events: " << n_written << "\n"
      << "Pythia seed: " << pythia_seed << "\n";

  if (n_written != n_events) {
    std::cerr
        << "ERROR: HepMC event count differs from request\n";
    return 5;
  }

  return 0;
}
