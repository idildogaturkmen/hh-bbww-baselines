# Finite-QCD-MC Reporting Policy

The exact central 68% Garwood interval uses gamma quantiles at 0.16 and 0.84. The exact one-sided 95% Poisson upper mean uses the 0.95 quantile of `Gamma(N+1,1)`. At `N=0`, `mu95 = 2.99573227355399`.

| observed QCD count | support label | recommended reporting rule |
|---:|---|---|
| N >= 1000 | `ADEQUATE_RAW_SUPPORT` | central with B95 cross-check |
| 100 <= N < 1000 | `FINITE_SUPPORT_CAUTION` | central with B95 cross-check |
| 20 <= N < 100 | `LIMITED` | central with B95 cross-check |
| 1 <= N < 20 | `EXTREMELY_LIMITED` | B95 is primary; central is diagnostic only |
| N = 0 | `ZERO_OBSERVED_MC` | B95 is primary; central is undefined |

At `N=0`, central `B_QCD`, `R_QCD`, `S/B`, `S/sqrt(B)`, and `Z_A` are `UNDEFINED`. Zero background, infinite rejection, and infinite significance are forbidden. The reported inference is `B_QCD_95UP` and `Z_A_USING_B_QCD_95UP_MCSTAT`.
