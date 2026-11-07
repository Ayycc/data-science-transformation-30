/*******************************************************************
 * 04_summarise.sas  Aggregate to the return grain and export to Excel
 * Output: \\fileshare\QRT\qrt_feed_&close_q..csv  (read by QRT_build.xlsm)
 *******************************************************************/
/* R08: SUM() in PROC SUMMARY ignores missing values */
proc summary data=work.prem_qtr nway;
  class quarter lob region;
  var written_cents earned_cents;
  output out=work.prem_sum(drop=_type_ _freq_) sum=;
run;

/* R05: policies in force = distinct policies with earned premium > 0 */
proc sort data=work.prem_qtr(where=(earned_cents > 0)) out=work.pif nodupkey;
  by quarter lob region policy_id;
run;
proc summary data=work.pif nway;
  class quarter lob region;
  output out=work.pif_sum(drop=_type_ rename=(_freq_=policies_in_force));
run;

proc summary data=work.clm_qtr nway;
  class quarter lob region;
  var paid_cents case_reserve_cents incurred_cents;
  output out=work.clm_sum(drop=_type_ _freq_) sum=;
run;

data work.qrt_feed;
  merge work.prem_sum work.clm_sum work.pif_sum;
  by quarter lob region;
  array n _numeric_;
  do over n; if n = . then n = 0; end;
run;

proc export data=work.qrt_feed outfile="\\fileshare\QRT\qrt_feed_&close_q..csv" dbms=csv replace; run;
