# Called immediately before VPL synthesis, including incremental resume.
set qwen_root [file dirname [file dirname [file normalize [info script]]]]
if {![file exists [file join $qwen_root scripts use_vendor_clock_adapter.py]]} {
    set qwen_root [pwd]
    while {![file exists [file join $qwen_root scripts use_vendor_clock_adapter.py]] && $qwen_root ne [file dirname $qwen_root]} {
        set qwen_root [file dirname $qwen_root]
    }
}
set qwen_vpl [pwd]
while {![file isdirectory [file join $qwen_vpl .local hw_platform]] && $qwen_vpl ne [file dirname $qwen_vpl]} {
    set qwen_vpl [file dirname $qwen_vpl]
}
if {![file isdirectory [file join $qwen_vpl .local hw_platform]]} {
    error "Cannot find generated VPL platform directory"
}
puts [exec /usr/bin/python3 -I [file join $qwen_root scripts use_vendor_clock_adapter.py] --build-root $qwen_vpl --apply]
# Retry only failed synthesis runs, preserving completed IP checkpoints.
foreach qwen_run [get_runs -quiet] {
    set qwen_status [get_property STATUS $qwen_run]
    if {[string match "*ERROR*" $qwen_status] || [string match "*Failed*" $qwen_status]} {
        puts "Reset failed run $qwen_run ($qwen_status)"
        reset_runs $qwen_run
    }
}
puts "VENDOR_CLOCK_ADAPTER_SELECTED"
