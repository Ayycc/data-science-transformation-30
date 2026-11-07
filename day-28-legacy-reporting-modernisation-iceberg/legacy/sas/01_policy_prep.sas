/*******************************************************************
 * 01_policy_prep.sas   Quarterly Regulatory Return - policy master
 * Owner: Actuarial MI   Last change: 2017-03-02 (JW)
 * Input : EXTRACT.POLICY_VERSIONS (nightly extract, file order = load_seq)
 * Output: WORK.POL_DEDUP  (one row per policy_id)
 *******************************************************************/
%let close_q = &SYSPARM;      /* 1..8 passed by the scheduler */

/* R01: one row per policy. NODUPKEY keeps the FIRST record per policy  */
/*      in input order. The extract is in load order, so this is        */
/*      normally version 1 (the original policy), but a 2025 backfill   */
/*      re-wrote some later versions ahead of version 1.                */
proc sort data=extract.policy_versions out=work.pol_sorted nodupkey;
  by policy_id;
run;

/* R02: line of business mapping. Codes must match exactly.             */
data work.pol_dedup;
  set work.pol_sorted;
  length lob $10;
  if lob_code_raw in ('COMMERCIAL','HOME','LIABILITY','MOTOR','TRAVEL') then lob = lob_code_raw;
  else lob = 'OTHER';
  keep policy_id lob region version_no;
run;
