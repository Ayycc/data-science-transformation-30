/*******************************************************************
 * 02_premium_earn.sas  Written and earned premium by quarter
 * Input : EXTRACT.PREMIUM_TXNS, WORK.POL_DEDUP
 * Output: WORK.PREM_QTR (one row per txn_id x quarter)
 *******************************************************************/
proc sort data=extract.premium_txns out=work.txn_sorted; by policy_id txn_id; run;

data work.prem_qtr;
  merge work.txn_sorted(in=a) work.pol_dedup(in=b);
  by policy_id;
  if a and b;
  array qs{8} _temporary_ ('01JAN2024'd '01APR2024'd '01JUL2024'd '01OCT2024'd
                           '01JAN2025'd '01APR2025'd '01JUL2025'd '01OCT2025'd);
  array qe{8} _temporary_ ('31MAR2024'd '30JUN2024'd '30SEP2024'd '31DEC2024'd
                           '31MAR2025'd '30JUN2025'd '30SEP2025'd '31DEC2025'd);
  term = txn_exp - txn_eff;                         /* txn_exp is exclusive */
  do quarter = 1 to &close_q;
    days = max(0, min(txn_exp, qe{quarter} + 1) - max(txn_eff, qs{quarter}));
    /* R03: written premium is recognised in the quarter of the txn effective date */
    if qs{quarter} <= txn_eff <= qe{quarter} then written_cents = premium_cents;
    else written_cents = 0;
    /* R04 + R11: pro-rata daily earning, ROUNDED TO THE CENT ON EVERY ROW */
    earned_cents = round(premium_cents * days / term, 1);
    if days > 0 or written_cents ne 0 then output;
  end;
  keep txn_id quarter policy_id lob region written_cents earned_cents;
run;
