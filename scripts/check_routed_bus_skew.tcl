# Read-only signoff report requested by Vivado's routed timing diagnostics.
if {$argc != 2} {
    puts stderr "Usage: check_routed_bus_skew.tcl routed.dcp report.rpt"
    exit 2
}
if {[catch {
    set_param general.maxThreads 2
    open_checkpoint [lindex $argv 0]
    report_bus_skew -file [lindex $argv 1]
    close_design
} detail]} {
    puts stderr "BUS_SKEW_REPORT_FAILED: $detail"
    exit 1
}
puts "BUS_SKEW_REPORT_WRITTEN"
exit
