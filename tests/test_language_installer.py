"""Test the language installation workflow."""

from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from odoo_data_flow.lib.actions.language_installer import (
    _install_languages_legacy,
    _install_languages_modern,
    _wait_for_languages_to_be_active,
    run_language_installation,
)


class TestLanguageInstaller:
    """Tests for the language installation action."""

    @patch("odoo_data_flow.lib.actions.language_installer.time.sleep")
    def test_wait_for_languages_success(self, mock_sleep: MagicMock) -> None:
        """Tests the success path of the language polling function."""
        mock_lang_model = MagicMock()
        mock_lang_model.search_read.side_effect = [
            [{"code": "de_DE"}],
            [{"code": "de_DE"}, {"code": "fr_FR"}],
        ]
        mock_connection = MagicMock()
        mock_connection.get_model.return_value = mock_lang_model

        result = _wait_for_languages_to_be_active(
            mock_connection, ["de_DE", "fr_FR"], timeout=10
        )

        assert result is True
        assert mock_lang_model.search_read.call_count == 2

    @patch("odoo_data_flow.lib.actions.language_installer.time.sleep")
    def test_wait_for_languages_timeout(self, mock_sleep: MagicMock) -> None:
        """Tests the timeout path of the language polling function."""
        mock_lang_model = MagicMock()
        mock_lang_model.search_read.return_value = [{"code": "de_DE"}]
        mock_connection = MagicMock()
        mock_connection.get_model.return_value = mock_lang_model

        result = _wait_for_languages_to_be_active(
            mock_connection, ["de_DE", "fr_FR"], timeout=1
        )
        assert result is False

    @patch("odoo_data_flow.lib.actions.language_installer.time.sleep")
    def test_wait_for_languages_rpc_error(self, mock_sleep: MagicMock) -> None:
        """Tests that the polling function handles an RPC error."""
        mock_lang_model = MagicMock()
        mock_lang_model.search_read.side_effect = Exception("RPC Error")
        mock_connection = MagicMock()
        mock_connection.get_model.return_value = mock_lang_model

        result = _wait_for_languages_to_be_active(mock_connection, ["de_DE"], timeout=1)
        assert result is False

    def test_install_languages_legacy_exception(self) -> None:
        """Test that _install_languages_legacy raises an exception."""
        mock_connection = MagicMock()
        mock_wizard_obj = MagicMock()
        mock_wizard_obj.create.side_effect = ValueError("Test Exception")
        mock_connection.get_model.return_value = mock_wizard_obj

        # with pytest.raises(Exception):
        with pytest.raises(ValueError):
            _install_languages_legacy(mock_connection, ["de_DE"])

    def test_install_languages_modern_fallback(self) -> None:
        """Test that _install_languages_modern falls back to 'langs' key."""
        mock_connection = MagicMock()
        mock_lang_model = MagicMock()
        mock_installer_model = MagicMock()
        mock_lang_model.search.return_value = [42]
        mock_installer_model.create.side_effect = [
            Exception("Invalid field lang_ids"),
            123,
        ]

        def get_model_side_effect(model_name: str) -> Any:
            if model_name == "res.lang":
                return mock_lang_model
            if model_name == "base.language.install":
                return mock_installer_model
            return MagicMock()

        mock_connection.get_model.side_effect = get_model_side_effect

        _install_languages_modern(mock_connection, ["de_DE"], 17)

        mock_installer_model.create.assert_has_calls(
            [
                call({"lang_ids": [(6, 0, [42])]}),
                call({"langs": [(6, 0, [42])]}),
            ]
        )

    @patch("odoo_data_flow.lib.odoo_lib.get_odoo_version")
    @patch("odoo_data_flow.lib.conf_lib.get_connection_from_config")
    @pytest.mark.parametrize(
        "version, expected_create_payload",
        [
            # Test case for Odoo < 15 (uses 'lang' key)
            (14, {"lang": "de_DE", "overwrite": False}),
            # FIX: Added 'overwrite': False to the expected payloads below
            # Test case for Odoo 15/16 (uses 'langs' key)
            (16, {"langs": [(6, 0, [42])], "overwrite": False}),
            # Test case for Odoo 17+ (uses 'lang_ids' key)
            (18, {"lang_ids": [(6, 0, [42])], "overwrite": False}),
        ],
    )
    def test_run_installation_for_all_versions(
        self,
        mock_get_conn: MagicMock,
        mock_get_version: MagicMock,
        version: int,
        expected_create_payload: dict[str, Any],
    ) -> None:
        """Tests the correct installation path for multiple Odoo versions."""
        # --- Arrange ---
        mock_get_version.return_value = version

        mock_lang_model = MagicMock()
        mock_installer_model = MagicMock()
        mock_lang_model.search.return_value = [42]
        mock_installer_model.create.return_value = 123

        def get_model_side_effect(model_name: str) -> Any:
            if model_name == "res.lang":
                return mock_lang_model
            if model_name == "base.language.install":
                return mock_installer_model
            return MagicMock()

        mock_get_conn.return_value.get_model.side_effect = get_model_side_effect

        # --- Act ---
        result = run_language_installation("dummy.conf", ["de_DE"])

        # --- Assert ---
        assert result is True
        mock_installer_model.create.assert_called_once_with(expected_create_payload)
        mock_installer_model.lang_install.assert_called_once_with([123])

    @patch("odoo_data_flow.lib.odoo_lib.get_odoo_version")
    @patch("odoo_data_flow.lib.conf_lib.get_connection_from_config")
    def test_installation_fails_if_language_not_found(
        self, mock_get_conn: MagicMock, mock_get_version: MagicMock
    ) -> None:
        """Test that the function returns False if a language code is invalid."""
        # --- Arrange ---
        # Force the modern (Odoo 15+) code path
        mock_get_version.return_value = 18

        mock_lang_model = MagicMock()
        # Simulate that the language code does not exist in Odoo
        mock_lang_model.search.return_value = []

        # The function now gets multiple models, so the mock needs to handle it
        def get_model_side_effect(model_name: str) -> Any:
            if model_name == "res.lang":
                return mock_lang_model
            # Return a generic mock for the installer model, which won't be used
            return MagicMock()

        mock_get_conn.return_value.get_model.side_effect = get_model_side_effect

        # --- Act ---
        result = run_language_installation("dummy.conf", ["xx_XX"])

        # --- Assert ---
        assert result is False

    @patch("odoo_data_flow.lib.conf_lib.get_connection_from_config")
    def test_installation_fails_gracefully_on_rpc_error(
        self, mock_get_conn: MagicMock
    ) -> None:
        """Test that the function returns False if an RPC error occurs."""
        mock_get_conn.side_effect = Exception("Connection refused")

        result = run_language_installation("dummy.conf", ["de_DE"])

        assert result is False

    @pytest.mark.parametrize("version", [14, 18])
    @patch("odoo_data_flow.lib.odoo_lib.get_odoo_version")
    @patch("odoo_data_flow.lib.conf_lib.get_connection_from_config")
    def test_run_installation_with_partial_failure(
        self,
        mock_get_conn: MagicMock,
        mock_get_version: MagicMock,
        version: int,
    ) -> None:
        """Test installation with partial failure.

        Tests that if one of several languages fails, the process continues
        but the final result is False.
        """
        # --- Arrange ---
        mock_get_version.return_value = version
        languages_to_install = [
            "de_DE",
            "fr_FR",
        ]  # One will succeed, one will fail

        mock_lang_model = MagicMock()
        mock_lang_model.search.return_value = [42]
        mock_installer_model = MagicMock()

        # Simulate create succeeding for de_DE but failing for fr_FR
        def create_side_effect(vals: dict[str, Any]) -> int:
            # Check for the modern payload structure
            if "lang_ids" in vals or "langs" in vals:
                # In modern versions, we can't easily know which lang this is for,
                # so we'll alternate the side effect based on call count.
                if mock_installer_model.create.call_count == 2:
                    raise Exception("RPC error on create for fr_FR")
            # Check for the legacy payload structure
            elif vals.get("lang") == "fr_FR":
                raise Exception("RPC error on create for fr_FR")
            return 123  # Success for de_DE

        mock_installer_model.create.side_effect = create_side_effect

        def get_model_side_effect(model_name: str) -> Any:
            return {
                "res.lang": mock_lang_model,
                "base.language.install": mock_installer_model,
            }.get(model_name)

        mock_get_conn.return_value.get_model.side_effect = get_model_side_effect

        # --- Act ---
        result = run_language_installation("dummy.conf", languages_to_install)

        # --- Assert ---
        assert result is False  # Overall result should be failure
        assert mock_installer_model.create.call_count == 2  # Both were attempted
        mock_installer_model.lang_install.assert_called_once_with(
            [123]
        )  # Only the successful one was executed

    @pytest.mark.parametrize("version", [14, 18])
    @patch("odoo_data_flow.lib.odoo_lib.get_odoo_version")
    @patch("odoo_data_flow.lib.conf_lib.get_connection_from_config")
    def test_run_installation_fails_on_install_step(
        self,
        mock_get_conn: MagicMock,
        mock_get_version: MagicMock,
        version: int,
    ) -> None:
        """Test run installation fails on install.

        Tests that a failure on the `lang_install` RPC call is handled correctly.
        """
        # --- Arrange ---
        mock_get_version.return_value = version
        mock_lang_model = MagicMock()
        mock_installer_model = MagicMock()

        # Simulate that create() works, but the lang_install() call fails
        mock_lang_model.search.return_value = [42]
        mock_installer_model.create.return_value = 123
        mock_installer_model.lang_install.side_effect = Exception("Execution error")

        def get_model_side_effect(model_name: str) -> Any:
            return {
                "res.lang": mock_lang_model,
                "base.language.install": mock_installer_model,
            }.get(model_name)

        mock_get_conn.return_value.get_model.side_effect = get_model_side_effect

        # --- Act ---
        result = run_language_installation("dummy.conf", ["de_DE"])

        # --- Assert ---
        assert result is False
        mock_installer_model.create.assert_called_once()
        mock_installer_model.lang_install.assert_called_once()

    def test_install_languages_legacy_success(self) -> None:
        """Test the success path for the legacy language installer."""
        mock_connection = MagicMock()
        mock_wizard_obj = MagicMock()
        mock_wizard_instance = MagicMock()

        # The create method returns a wizard ID
        mock_wizard_obj.create.side_effect = [123, 124]
        # The browse method returns a mock that can have lang_install called on it
        mock_wizard_obj.browse.return_value = mock_wizard_instance
        mock_connection.get_model.return_value = mock_wizard_obj

        languages = ["de_DE", "fr_FR"]
        _install_languages_legacy(mock_connection, languages)

        # Check that create was called for each language
        expected_create_calls = [
            call({"lang": "de_DE", "overwrite": False}),
            call({"lang": "fr_FR", "overwrite": False}),
        ]
        mock_wizard_obj.create.assert_has_calls(expected_create_calls)
        assert mock_wizard_obj.create.call_count == 2

        # --- FIX START ---
        # Check that browse was called with each wizard ID
        mock_wizard_obj.browse.assert_any_call(123)
        mock_wizard_obj.browse.assert_any_call(124)
        # Ensure browse was called exactly twice
        assert mock_wizard_obj.browse.call_count == 2

        # Check that lang_install was called for each created wizard
        assert mock_wizard_instance.lang_install.call_count == 2
        # --- FIX END ---

    # --- NEW TEST to cover lines 55-58 (language not found) ---
    def test_install_languages_modern_no_lang_found(self) -> None:
        """Test modern installer when no language IDs are found."""
        mock_connection = MagicMock()
        mock_lang_model = MagicMock()
        mock_installer_model = MagicMock()

        # Simulate search returning no IDs
        mock_lang_model.search.return_value = []

        def get_model_side_effect(model_name: str) -> Any:
            if model_name == "res.lang":
                return mock_lang_model
            if model_name == "base.language.install":
                return mock_installer_model
            return MagicMock()

        mock_connection.get_model.side_effect = get_model_side_effect

        _install_languages_modern(mock_connection, ["xx_XX"], 17)

        # Assert that the installer wizard was never created
        mock_installer_model.create.assert_not_called()

    # --- NEW TEST to cover lines 78-80 (fallback attempt also fails) ---
    def test_install_languages_modern_fallback_fails(self) -> None:
        """Test that the installer raises if the fallback also fails."""
        mock_connection = MagicMock()
        mock_lang_model = MagicMock()
        mock_installer_model = MagicMock()
        mock_lang_model.search.return_value = [42]
        # Simulate create failing for both 'lang_ids' and the 'langs' fallback
        mock_installer_model.create.side_effect = [
            ValueError("Invalid field lang_ids"),
            ValueError("Invalid field langs"),
        ]

        def get_model_side_effect(model_name: str) -> Any:
            if model_name == "res.lang":
                return mock_lang_model
            if model_name == "base.language.install":
                return mock_installer_model
            return MagicMock()

        mock_connection.get_model.side_effect = get_model_side_effect

        with pytest.raises(ValueError, match="Invalid field langs"):
            _install_languages_modern(mock_connection, ["de_DE"], 17)

        assert mock_installer_model.create.call_count == 2

    # --- NEW TEST to cover lines 81-82 (non-fallback key fails) ---
    def test_install_languages_modern_legacy_key_fails(self) -> None:
        """Test that an error with a non-fallback key is re-raised."""
        mock_connection = MagicMock()
        mock_lang_model = MagicMock()
        mock_installer_model = MagicMock()
        mock_lang_model.search.return_value = [42]
        mock_installer_model.create.side_effect = ValueError("Create error")

        def get_model_side_effect(model_name: str) -> Any:
            if model_name == "res.lang":
                return mock_lang_model
            if model_name == "base.language.install":
                return mock_installer_model
            return MagicMock()

        mock_connection.get_model.side_effect = get_model_side_effect

        # Use version 16, so the initial key is 'langs'
        with pytest.raises(ValueError, match="Create error"):
            _install_languages_modern(mock_connection, ["de_DE"], 16)

        # Assert create was only called once, as no fallback is attempted
        mock_installer_model.create.assert_called_once_with({"langs": [(6, 0, [42])]})

    @patch("odoo_data_flow.lib.actions.language_installer.log")
    def test_install_languages_modern_failure_on_install_step(
        self, mock_log: MagicMock
    ) -> None:
        """Tests that a failure on the `lang_install` RPC call is handled.

        This test specifically ensures the `except` block in `_install_languages_modern`
        is entered when the installation step fails, covering the log statement
        within it.
        """
        # ARRANGE
        mock_connection = MagicMock()
        mock_lang_model = MagicMock()
        mock_installer_model = MagicMock()
        mock_wizard_instance = MagicMock()

        mock_lang_model.search.return_value = [42]

        mock_installer_model.create.side_effect = [
            123,
            ValueError("Fallback create failed"),
        ]

        mock_installer_model.browse.return_value = mock_wizard_instance
        install_exception = RuntimeError("RPC error during lang_install")
        mock_wizard_instance.lang_install.side_effect = install_exception

        def get_model_side_effect(model_name: str) -> Any:
            model_map = {
                "res.lang": mock_lang_model,
                "base.language.install": mock_installer_model,
            }
            return model_map.get(model_name, MagicMock())

        mock_connection.get_model.side_effect = get_model_side_effect

        # ACT & ASSERT
        with pytest.raises(ValueError, match="Fallback create failed"):
            _install_languages_modern(mock_connection, ["de_DE"], 17)

        # --- FIX START ---
        # ASSERT that both error log calls were made in the correct order.
        expected_log_calls = [
            call(
                "Failed to create language wizard with key 'lang_ids': "
                "RPC error during lang_install"
            ),
            call("Fallback attempt also failed: Fallback create failed"),
        ]
        mock_log.error.assert_has_calls(expected_log_calls)
        assert mock_log.error.call_count == 2
        # --- FIX END ---

        # Verify the sequence of other mock calls.
        assert mock_installer_model.create.call_count == 2
        mock_installer_model.browse.assert_called_once_with(123)
        mock_wizard_instance.lang_install.assert_called_once()
