# Synthesize a vendor-supplied clock adapter in isolation, without board access.
if {$argc != 1} {
    puts stderr "Usage: check_clock_metadata.tcl vendor_source.v"
    exit 2
}
if {[catch {
    create_project -in_memory -part xcu55c-fsvh2892-2L-e
    set_param general.maxThreads 2
    read_verilog [lindex $argv 0]
    synth_design -mode out_of_context -top clk_metadata_adapter_v1_0_0 -part xcu55c-fsvh2892-2L-e
    close_project
} detail]} {
    puts stderr "CLOCK_METADATA_SYNTH_FAILED: $detail"
    exit 1
}
puts "CLOCK_METADATA_SYNTH_PASSED"
exit
