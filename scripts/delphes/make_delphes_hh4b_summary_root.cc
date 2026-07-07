#include "classes/DelphesClasses.h"

#include "TClonesArray.h"
#include "TFile.h"
#include "TTree.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct SelectedJet {
  double pt;
  double eta;
  double phi;
  double mass;
  int btag;
  int flavor;
};

struct PairingResult {
  std::string pairing;
  double mbb1;
  double mbb2;
  double score;
  double drbb1;
  double drbb2;
};

double delta_phi(double a, double b) {
  return std::atan2(std::sin(a - b), std::cos(a - b));
}

double delta_r(const SelectedJet& a, const SelectedJet& b) {
  const double deta = a.eta - b.eta;
  const double dphi = delta_phi(a.phi, b.phi);
  return std::sqrt(deta * deta + dphi * dphi);
}

void four_vector(
    const SelectedJet& jet, double& e, double& px, double& py, double& pz) {
  px = jet.pt * std::cos(jet.phi);
  py = jet.pt * std::sin(jet.phi);
  pz = jet.pt * std::sinh(jet.eta);
  e = std::sqrt(px * px + py * py + pz * pz + jet.mass * jet.mass);
}

double invariant_mass(const SelectedJet& a, const SelectedJet& b) {
  double e1, px1, py1, pz1;
  double e2, px2, py2, pz2;
  four_vector(a, e1, px1, py1, pz1);
  four_vector(b, e2, px2, py2, pz2);

  const double e = e1 + e2;
  const double px = px1 + px2;
  const double py = py1 + py2;
  const double pz = pz1 + pz2;
  const double m2 = e * e - px * px - py * py - pz * pz;
  return std::sqrt(std::max(m2, 0.0));
}

double m4(const std::vector<SelectedJet>& jets) {
  double e_sum = 0.0;
  double px_sum = 0.0;
  double py_sum = 0.0;
  double pz_sum = 0.0;
  for (const auto& jet : jets) {
    double e, px, py, pz;
    four_vector(jet, e, px, py, pz);
    e_sum += e;
    px_sum += px;
    py_sum += py;
    pz_sum += pz;
  }
  const double m2 = e_sum * e_sum - px_sum * px_sum - py_sum * py_sum - pz_sum * pz_sum;
  return std::sqrt(std::max(m2, 0.0));
}

PairingResult best_pairing(const std::vector<SelectedJet>& jets4, double target_mass) {
  const int pairings[3][4] = {
      {0, 1, 2, 3},
      {0, 2, 1, 3},
      {0, 3, 1, 2},
  };
  const char* labels[3] = {
      "((0, 1), (2, 3))",
      "((0, 2), (1, 3))",
      "((0, 3), (1, 2))",
  };

  PairingResult best;
  best.score = 1.0e99;

  for (int ip = 0; ip < 3; ++ip) {
    const auto& a = jets4[pairings[ip][0]];
    const auto& b = jets4[pairings[ip][1]];
    const auto& c = jets4[pairings[ip][2]];
    const auto& d = jets4[pairings[ip][3]];
    const double mbb1 = invariant_mass(a, b);
    const double mbb2 = invariant_mass(c, d);
    const double score = std::abs(mbb1 - target_mass) + std::abs(mbb2 - target_mass);
    if (score < best.score) {
      best.pairing = labels[ip];
      best.mbb1 = mbb1;
      best.mbb2 = mbb2;
      best.score = score;
      best.drbb1 = delta_r(a, b);
      best.drbb2 = delta_r(c, d);
    }
  }
  return best;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " input.root event_summary.csv candidates.csv sample\n";
    return 1;
  }

  const std::string input = argv[1];
  const std::string event_csv = argv[2];
  const std::string candidate_csv = argv[3];
  const std::string sample = argv[4];

  TFile file(input.c_str());
  if (file.IsZombie()) {
    std::cerr << "ERROR: cannot open " << input << "\n";
    return 2;
  }

  auto* tree = dynamic_cast<TTree*>(file.Get("Delphes"));
  if (!tree) {
    std::cerr << "ERROR: missing Delphes tree in " << input << "\n";
    return 3;
  }

  TClonesArray* event_branch = nullptr;
  TClonesArray* particle_branch = nullptr;
  TClonesArray* jet_branch = nullptr;
  TClonesArray* genjet_branch = nullptr;
  TClonesArray* electron_branch = nullptr;
  TClonesArray* muon_branch = nullptr;
  TClonesArray* met_branch = nullptr;

  tree->SetBranchAddress("Event", &event_branch);
  tree->SetBranchAddress("Particle", &particle_branch);
  tree->SetBranchAddress("Jet", &jet_branch);
  tree->SetBranchAddress("GenJet", &genjet_branch);
  tree->SetBranchAddress("Electron", &electron_branch);
  tree->SetBranchAddress("Muon", &muon_branch);
  tree->SetBranchAddress("MissingET", &met_branch);

  std::ofstream events(event_csv);
  std::ofstream candidates(candidate_csv);
  events << std::setprecision(10);
  candidates << std::setprecision(10);

  events << "sample,event,event_weight,event_cross_section_pb,n_particle,n_jet,"
         << "n_jet_pt30_eta25,n_bjet_pt30_eta25,n_genjet,n_electron,n_muon,"
         << "ht_pt30_eta25,met,n_higgs_particles,n_b_particles\n";
  candidates << "sample,event,n_selected_bjets,mbb1,mbb2,avg_mbb,delta_mbb,mhh,"
             << "drbb1,drbb2,pairing,j1_pt,j2_pt,j3_pt,j4_pt\n";

  const Long64_t n_events = tree->GetEntries();
  Long64_t n_candidates = 0;

  for (Long64_t iev = 0; iev < n_events; ++iev) {
    tree->GetEntry(iev);

    double event_weight = 1.0;
    double event_cross_section_pb = NAN;
    if (event_branch && event_branch->GetEntries() > 0) {
      if (auto* event = dynamic_cast<HepMCEvent*>(event_branch->At(0))) {
        event_weight = event->Weight;
        event_cross_section_pb = event->CrossSection;
      } else if (auto* event = dynamic_cast<LHEFEvent*>(event_branch->At(0))) {
        event_weight = event->Weight;
        event_cross_section_pb = event->CrossSection;
      }
    }

    int n_higgs = 0;
    int n_b = 0;
    if (particle_branch) {
      for (int ip = 0; ip < particle_branch->GetEntries(); ++ip) {
        auto* particle = static_cast<GenParticle*>(particle_branch->At(ip));
        if (std::abs(particle->PID) == 25) ++n_higgs;
        if (std::abs(particle->PID) == 5) ++n_b;
      }
    }

    int n_selected_jets = 0;
    int n_selected_bjets = 0;
    double ht = 0.0;
    std::vector<SelectedJet> selected_bjets;

    if (jet_branch) {
      for (int ij = 0; ij < jet_branch->GetEntries(); ++ij) {
        auto* jet = static_cast<Jet*>(jet_branch->At(ij));
        const bool selected = jet->PT > 30.0 && std::abs(jet->Eta) < 2.5;
        if (!selected) continue;
        ++n_selected_jets;
        ht += jet->PT;
        if (jet->BTag > 0) {
          ++n_selected_bjets;
          selected_bjets.push_back({
              jet->PT,
              jet->Eta,
              jet->Phi,
              jet->Mass,
              static_cast<int>(jet->BTag),
              static_cast<int>(jet->Flavor),
          });
        }
      }
    }

    double met = 0.0;
    if (met_branch && met_branch->GetEntries() > 0) {
      met = static_cast<MissingET*>(met_branch->At(0))->MET;
    }

    events << sample << "," << iev << "," << event_weight << ","
           << event_cross_section_pb << ","
           << (particle_branch ? particle_branch->GetEntries() : 0) << ","
           << (jet_branch ? jet_branch->GetEntries() : 0) << ","
           << n_selected_jets << "," << n_selected_bjets << ","
           << (genjet_branch ? genjet_branch->GetEntries() : 0) << ","
           << (electron_branch ? electron_branch->GetEntries() : 0) << ","
           << (muon_branch ? muon_branch->GetEntries() : 0) << ","
           << ht << "," << met << "," << n_higgs << "," << n_b << "\n";

    std::sort(
        selected_bjets.begin(),
        selected_bjets.end(),
        [](const SelectedJet& a, const SelectedJet& b) { return a.pt > b.pt; });

    if (selected_bjets.size() < 4) continue;

    std::vector<SelectedJet> jets4(
        selected_bjets.begin(), selected_bjets.begin() + 4);
    const auto best = best_pairing(jets4, 125.0);
    const double avg_mbb = 0.5 * (best.mbb1 + best.mbb2);
    const double delta_mbb = std::abs(best.mbb1 - best.mbb2);

    candidates << sample << "," << iev << "," << selected_bjets.size() << ","
               << best.mbb1 << "," << best.mbb2 << "," << avg_mbb << ","
               << delta_mbb << "," << m4(jets4) << "," << best.drbb1 << ","
               << best.drbb2 << ",\"" << best.pairing << "\","
               << jets4[0].pt << "," << jets4[1].pt << "," << jets4[2].pt
               << "," << jets4[3].pt << "\n";
    ++n_candidates;
  }

  std::cout << "Input events: " << n_events << "\n";
  std::cout << "Candidate events: " << n_candidates << "\n";
  std::cout << "Wrote: " << event_csv << "\n";
  std::cout << "Wrote: " << candidate_csv << "\n";
  return 0;
}
