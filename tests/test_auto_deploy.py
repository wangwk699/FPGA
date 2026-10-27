import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import auto_deploy


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        self.patches = [patch.object(auto_deploy, "REPORTS", self.path),
                        patch.object(auto_deploy, "STATE", self.path / "status.json")]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def test_failed_command_never_records_step_as_complete(self):
        pipeline = auto_deploy.Pipeline(os.getpid())
        with self.assertRaisesRegex(RuntimeError, "exit code 3"):
            pipeline.command("failing_step", [sys.executable, "-c", "raise SystemExit(3)"], 10)
        self.assertEqual(pipeline.state["completed_steps"], [])
        self.assertFalse(pipeline.state["fpga_deployment_complete"])

    def test_successful_command_records_log_and_completion(self):
        pipeline = auto_deploy.Pipeline(os.getpid())
        pipeline.command("probe", [sys.executable, "-c", "print('probe complete')"], 10)
        state = json.loads((self.path / "status.json").read_text())
        self.assertEqual(state["completed_steps"], ["probe"])
        self.assertIn("probe complete", (self.path / "auto-probe.log").read_text())
        self.assertFalse(state["fpga_deployment_complete"])

    def test_progress_uses_last_carriage_return_update(self):
        (self.path / "vitis-install-resume.log").write_text(
            "INFO - Started\n\rDownloading files (1 MB / 45 GB)\rDownloading files (2 MB / 45 GB)")
        self.assertEqual(auto_deploy.download_progress(), "Downloading files (2 MB / 45 GB)")

    def test_progress_does_not_include_appended_error_message(self):
        (self.path / "vitis-install-resume.log").write_text(
            "Downloading files (10 GB / 45 GB) -INFO  - Canceling. Please wait...")
        self.assertEqual(auto_deploy.download_progress(), "Downloading files (10 GB / 45 GB) -")

    def test_tls_failure_is_retryable_without_exposing_signed_url(self):
        failure = auto_deploy.classify_install_failure(
            "ERROR - Download failed. Checksum failed https://example.test/file?secret=value",
            "javax.net.ssl.SSLHandshakeException: Remote host terminated the handshake")
        self.assertEqual(failure["kind"], "download_network")
        self.assertTrue(failure["retryable"])
        self.assertNotIn("secret", failure["summary"])

    def test_auth_disk_and_cancel_failures_are_not_retried(self):
        for message, expected in [
            ("Before installing you must generate an authentication token", "authentication"),
            ("No space left on device", "disk_space"),
            ("INFO - Canceling. Please wait...", "cancelled"),
        ]:
            with self.subTest(expected=expected):
                failure = auto_deploy.classify_install_failure(message)
                self.assertEqual(failure["kind"], expected)
                self.assertFalse(failure["retryable"])

    def test_owned_installer_retries_network_failure_and_records_exit_status(self):
        counter = self.path / "attempts.txt"
        code = ("from pathlib import Path; import sys; p=Path(sys.argv[1]); "
                "n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1)); "
                "print('ERROR - Download failed' if n==0 else 'INFO - Installation completed successfully.'); "
                "raise SystemExit(1 if n==0 else 0)")
        pipeline = auto_deploy.Pipeline(None)
        pipeline.install_tools(max_attempts=2, retry_delay=0,
            command=[sys.executable, "-c", code, str(counter)])
        self.assertEqual(counter.read_text(), "2")
        self.assertEqual(pipeline.state["installer_exit_code"], 0)
        self.assertEqual(pipeline.state["completed_steps"], ["install_tools"])

    def test_owned_installer_stops_on_authentication_failure(self):
        pipeline = auto_deploy.Pipeline(None)
        with self.assertRaisesRegex(RuntimeError, "authentication"):
            pipeline.install_tools(max_attempts=3, retry_delay=0, command=[sys.executable, "-c",
                "print('ERROR - You must generate an authentication token'); raise SystemExit(1)"])
        self.assertEqual(pipeline.state["install_attempt"], 1)
        self.assertEqual(pipeline.state["completed_steps"], [])

    def test_attached_failed_installer_reports_download_failure_before_tool_search(self):
        (self.path / "vitis-install-resume.log").write_text(
            "ERROR - Download failed\nERROR - The installation failed")
        pipeline = auto_deploy.Pipeline(1073741824)
        with self.assertRaisesRegex(RuntimeError, "download failed"):
            pipeline.wait_for_installer(1073741824)
        self.assertEqual(pipeline.state["stage"], "installing_tools")

    def test_process_identity_detects_running_and_missing_processes(self):
        self.assertIsNotNone(auto_deploy.process_identity(os.getpid()))
        self.assertIsNone(auto_deploy.process_identity(1073741824))


if __name__ == "__main__":
    unittest.main()
