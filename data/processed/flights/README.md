# Avinor processed data

- AIRPORT – Airport for departure or arrival (OSL or TRD)
- DEP_ARR – Departure or arrival
- AIRLINE_IATA – IATA-code for airline
- FLIGHT_ID – Flight number

- TO_FROM – Destination (DEPartures) or origin-airport (ARRivals) – IATA-code for airport
- INT_DOM_SCHENGEN – D = domestic , I = International (Non-Schengen), S = Schengen
- GATE_STAND – For departures "gate" is used, while "stand" is used for arrivals.
- CANCELLED – (value "C" if cancelled). Note that airlines decide if they want to mark a flight as "cancelled" or a "reroute" or "replace". Most flights that's cancelled last minute (same day) gets marked as cancelled. While COVID-flights mostly are marked as a replace, which just deletes the flight from Avinor's system. However, this i not always the case so this data field has low value in determining Covid vs non-Covid cancellations.
- **SOBT_SIBT** - Scheduled route time (local time)
  - SOBT - Scheduled Off-Block Time
  - SIBT - Scheduled In-Block Time
- **ATOT_ALDT** - Actual takeoff/landing time on runway (local time)

  - ATOT - Actual Take Off Time
  - ALDT - Actual Landing Time

- **AOBT_AIBT** - Actual departure or arrival time at gate (local time)
  - AOBT - Actual Off-Block Time
  - AIBT - Actual In-Block Time
- **WEEKDAY** - 0=Monday, 6=Sunday
- **DELAY** - (AOBT_AIBT - SOBT_SIBT), gives positive number for positive delay (behind schedule) and negative for before schedule). If a flight is marked as cancelled, this field will equal "Inf".
