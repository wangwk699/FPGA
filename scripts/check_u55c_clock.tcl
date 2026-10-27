# Isolated installation check; does not access or program the FPGA card.
if {[catch {
    set check_dir [file normalize [file join [pwd] build clock_check_[pid]]]
    create_project clock_check $check_dir -part xcu55c-fsvh2892-2L-e
    create_bd_design clock_check
    create_bd_cell -vlnv xilinx.com:ip:clk_wiz:6.0 -type ip clkwiz_hbm
    close_project
} detail]} {
    puts stderr "U55C_CLOCK_IP_FAILED: $detail"
    exit 1
}
puts "U55C_CLOCK_IP_CREATED"
exit
