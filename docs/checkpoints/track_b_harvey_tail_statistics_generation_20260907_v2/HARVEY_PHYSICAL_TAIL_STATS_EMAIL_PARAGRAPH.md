The tightest background regions we can currently claim are thinly but not
single-event supported: at score >= 0.9997 there are 100 raw background
events (weighted B = 532.4 at 450 fb-1, effective N = 81.0, 11.1% relative
MC uncertainty), and at score >= 0.99997 there are 15 raw events (B = 97.3,
effective N = 14.0, 26.7% relative uncertainty) -- QCD supplies 84-90% of
the weighted background in both regions, with ttbar and fourteen minor
processes making up the rest. At our existing ~4% signal-efficiency working
point specifically, the correctly weighted uncertainty is 22.1% (not the
20.9% a naive raw-count estimate would give), because a handful of
higher-weight processes make the effective statistics somewhat worse than
the raw event count alone suggests. Generating 4x today's QCD Monte Carlo,
at the same physical cross-section and luminosity, would bring that down to
roughly 11.9% (and to 14.4% in the 0.99997 region) -- but it would cost
about 22.6 million CPU-slot-hours, which is on the order of a year even at
5,000 concurrently running slots, and none of that compute buys any change
in the underlying physical background yield, only better precision on it.

Brute-force scaling is therefore expensive for the precision it returns.
A targeted extension -- generating additional QCD preferentially in the
high-pTHat region that our existing statistics already suggest is where the
tail-feeding events concentrate -- is very likely a better use of compute,
using Pythia's native, exact-compensating-weight event-biasing mechanism so
the total cross-section stays unbiased by construction. That said, this is
not yet a validated result: it requires carrying a genuine per-event weight
through every downstream calculation instead of today's flat per-process
weight, and it must pass a small mechanism-validation run (roughly 20,000
events, well under an hour of compute) demonstrating that the biased run's
weighted spectrum still matches our existing sample before any larger
campaign is requested. We'd recommend running that validation step next,
rather than committing to either a large brute-force regeneration or a
full-scale targeted campaign.
