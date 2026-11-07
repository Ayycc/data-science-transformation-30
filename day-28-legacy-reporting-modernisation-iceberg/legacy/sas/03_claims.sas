/*******************************************************************
 * 03_claims.sas  Claim movements by valuation quarter
 * Input : EXTRACT.CLAIM_VALUATIONS (cumulative paid, case reserve)
 * Output: WORK.CLM_QTR (one row per claim_id x valuation quarter)
 *******************************************************************/
proc sort data=extract.claim_valuations out=work.val_sorted; by policy_id; run;

data work.clm_join;
  merge work.val_sorted(in=a) work.pol_dedup(in=b);
  by policy_id;
  if a and b;
run;

proc sort data=work.clm_join; by claim_id valuation_q; run;

data work.clm_qtr;
  set work.clm_join;
  by claim_id;
  retain prev_paid prev_inc;
  if first.claim_id then do; prev_paid = 0; prev_inc = 0; end;
  /* R07: '+' propagates missing. A missing case reserve (valuation    */
  /*      system outage) makes incurred missing, so the paid amount    */
  /*      silently drops out of incurred for this and the next quarter. */
  incurred_cum_cents = cum_paid_cents + case_reserve_cents;
  /* R06: quarterly paid movement */
  paid_cents      = cum_paid_cents - prev_paid;
  incurred_cents  = incurred_cum_cents - prev_inc;
  prev_paid = cum_paid_cents;
  prev_inc  = incurred_cum_cents;
  quarter = valuation_q;
  keep claim_id quarter policy_id lob region paid_cents case_reserve_cents incurred_cum_cents incurred_cents;
run;
