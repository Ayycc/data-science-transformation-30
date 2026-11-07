Attribute VB_Name = "QRT_build"
' ==================================================================
' QRT_build.xlsm   Builds the "Return" sheet from qrt_feed_<q>.csv
' Owner: Finance Reporting   Last change: 2019-11-18 (MK)
' Sheet "ELR" holds the expected loss ratio table A2:B6, sorted A-Z:
'   COMMERCIAL 0.62 | HOME 0.58 | LIABILITY 0.74 | MOTOR 0.71 | TRAVEL 0.45
' ==================================================================
Option Explicit

Sub BuildReturn()
    Dim ws As Worksheet, lastRow As Long, q As Integer
    Set ws = ThisWorkbook.Sheets("Return")
    q = ThisWorkbook.Sheets("Control").Range("B1").Value          ' close quarter 1..8
    ws.Range("A2:L50000").ClearContents
    ImportFeed ws, "\\fileshare\QRT\qrt_feed_" & q & ".csv"
    lastRow = ws.Cells(ws.Rows.Count, 1).End(xlUp).Row

    ' Columns: A quarter B lob C region D written E earned_cents F paid G reserve H incurred
    ' R11: earned premium in the return is ROUNDed to the cent (half away from zero)
    ws.Range("E2:E" & lastRow).Formula = "=ROUND(qrt_feed!E2,0)"
    ' R09: loss ratio, 4 dp; IFERROR turns a zero-earned cell into 0%
    ws.Range("I2:I" & lastRow).Formula = "=IFERROR(ROUND(H2/E2,4),0)"
    ' R10: IBNR (required from 2025Q1). VLOOKUP with TRUE = approximate match:
    '      a code missing from the ELR sheet takes the row above it alphabetically.
    If q >= 5 Then
        ws.Range("J2:J" & lastRow).Formula = "=MAX(0,ROUND(E2*VLOOKUP(B2,ELR!$A$2:$B$6,2,TRUE),0)-H2)"
    End If
    ws.Range("K2:K" & lastRow).Formula = "=qrt_feed!K2"                ' policies in force
    ws.Calculate
    ws.Range("A1:K" & lastRow).Value = ws.Range("A1:K" & lastRow).Value  ' paste values before filing
End Sub

Private Sub ImportFeed(ws As Worksheet, path As String)
    ' copies the CSV into hidden sheet qrt_feed (omitted)
End Sub
