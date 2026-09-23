# Mathematical model

For (N) identical FCFS servers, total Poisson demand (N\lambda), exponential
service rate \(\mu\), and JSQ(\(d\)), let \(q_k\) be the fraction of queues of
length at least \(k\). The mean-field equation is

\[
\dot q_k=\lambda(q_{k-1}^d-q_k^d)-\mu(q_k-q_{k+1}),\qquad q_0=1.
\]

At stationarity, write \(\rho=\lambda/\mu\). Telescoping the equilibrium
flow balance gives \(q_{k+1}=\rho q_k^d\). Starting from \(q_0=1\), this yields

\[
q_k=\rho^{(d^k-1)/(d-1)}\quad(d\ge2),\qquad q_k=\rho^k\quad(d=1).
\]

The implementation checks the original stationary recurrence numerically and
truncates after the first \(q_k<10^{-12}\), subject to a configured safety limit.
Expected jobs per server are \(L=\sum_{k\ge1}q_k\); Little's law gives system
time \(W=L/\lambda\). At zero demand the limiting system time is \(1/\mu\).

For control, a candidate capacity \(m\) uses per-server load
\(\rho=\Lambda/(m\mu)\) and the stationary fixed point locally. This is a
quasi-stationary approximation, not a transient mean-field solution. Loads at
or above the configured \(\rho_{max}\) receive a large finite penalty.

