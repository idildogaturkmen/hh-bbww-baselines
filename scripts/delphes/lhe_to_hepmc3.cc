#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/WriterAscii.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " input.lhe output.hepmc nEvents\n";
    return 1;
  }

  const std::string lhe_file = argv[1];
  const std::string hepmc_file = argv[2];
  const int n_events = std::atoi(argv[3]);

  Pythia8::Pythia pythia;

  pythia.readString("Beams:frameType = 4");
  pythia.readString("Beams:LHEF = " + lhe_file);

  pythia.readString("PartonLevel:all = on");
  pythia.readString("PartonLevel:ISR = on");
  pythia.readString("PartonLevel:FSR = on");
  pythia.readString("HadronLevel:all = on");

  pythia.readString("Check:event = off");

  pythia.readString("Next:numberCount = 10");
  pythia.readString("Next:numberShowInfo = 1");
  pythia.readString("Next:numberShowProcess = 1");
  pythia.readString("Next:numberShowEvent = 1");

  if (!pythia.init()) {
    std::cerr << "ERROR: Pythia initialization failed\n";
    return 2;
  }

  HepMC3::WriterAscii hepmc_writer(hepmc_file);
  Pythia8::Pythia8ToHepMC to_hepmc;

  int n_written = 0;

  for (int i = 0; i < n_events; ++i) {
    if (!pythia.next()) {
      std::cerr << "WARNING: pythia.next() failed at event " << i << "\n";
      continue;
    }

    HepMC3::GenEvent hepmc_event;
    to_hepmc.fill_next_event(pythia, &hepmc_event);
    hepmc_writer.write_event(hepmc_event);

    ++n_written;
  }

  pythia.stat();
  hepmc_writer.close();

  std::cout << "Wrote " << n_written << " events to " << hepmc_file << "\n";
  return 0;
}
