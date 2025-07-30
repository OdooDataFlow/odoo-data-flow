"""Test the language installation workflow."""

from unittest.mock import MagicMock, call, patch

import pytest

from odoo_data_flow.lib.actions.language_installer import (
    _wait_for_languages_to_be_active,
    run_language_installation,
)


class TestLanguageInstaller:
    """Tests for the language installation workflow."""

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

    @patch(
        "odoo_data_flow.lib.actions.language_installer._wait_for_languages_to_be_active",
        return_value=True,
    )
    @patch("odoo_data_flow.lib.actions.language_installer.odoo_lib.get_odoo_version")
    @patch(
        "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
    )
    def test_run_installation_legacy_success(
        self,
        mock_get_connection: MagicMock,
        mock_get_odoo_version: MagicMock,
        mock_wait: MagicMock,
    ) -> None:
        """Test successful language installation for legacy Odoo versions (<15)."""
        mock_get_odoo_version.return_value = 14
        mock_wizard_obj = MagicMock()
        mock_wizard_obj.create.return_value = 123
        mock_connection = MagicMock()
        mock_connection.get_model.return_value = mock_wizard_obj
        mock_get_connection.return_value = mock_connection

        run_language_installation(config="dummy.conf", languages=["nl_BE", "fr_FR"])

        mock_connection.get_model.assert_called_once_with("base.language.install")
        expected_create_calls = [
            call({"lang": "nl_BE"}),
            call({"lang": "fr_FR"}),
        ]
        mock_wizard_obj.create.assert_has_calls(expected_create_calls, any_order=True)
        mock_wizard_obj.browse(123).lang_install.assert_called()
        mock_wait.assert_called_once()

    @patch(
        "odoo_data_flow.lib.actions.language_installer._wait_for_languages_to_be_active",
        return_value=True,
    )
    @patch("odoo_data_flow.lib.actions.language_installer.odoo_lib.get_odoo_version")
    @patch(
        "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
    )
    def test_run_installation_v18_success(
        self,
        mock_get_connection: MagicMock,
        mock_get_odoo_version: MagicMock,
        mock_wait: MagicMock,
    ) -> None:
        """Tests the successful installation path for Odoo 18+."""
        mock_get_odoo_version.return_value = 18
        mock_wizard_obj = MagicMock()
        mock_wizard_obj.create.return_value = 42
        mock_connection = MagicMock()
        mock_connection.get_model.return_value = mock_wizard_obj
        mock_get_connection.return_value = mock_connection

        run_language_installation(config="dummy.conf", languages=["de_DE"])

        mock_connection.get_model.assert_called_once_with("base.language.install")
        mock_wizard_obj.create.assert_called_once_with({"lang": "de_DE"})
        mock_wizard_obj.browse.assert_called_once_with(42)
        mock_wizard_obj.browse.return_value.lang_install.assert_called_once()
        mock_wait.assert_called_once()

    @pytest.mark.parametrize(
        "version, expected_key",
        [(15, "langs"), (16, "langs"), (17, "lang_ids")],
    )
    @patch(
        "odoo_data_flow.lib.actions.language_installer._wait_for_languages_to_be_active",
        return_value=True,
    )
    @patch("odoo_data_flow.lib.actions.language_installer.odoo_lib.get_odoo_version")
    @patch(
        "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
    )
    def test_run_installation_v15_to_v17_success(
        self,
        mock_get_connection: MagicMock,
        mock_get_odoo_version: MagicMock,
        mock_wait: MagicMock,
        version: int,
        expected_key: str,
    ) -> None:
        """Tests the successful installation path for Odoo 15, 16, and 17."""
        mock_get_odoo_version.return_value = version
        mock_lang_model = MagicMock()
        mock_lang_model.search.return_value = [101, 102]
        mock_wizard_obj = MagicMock()
        mock_wizard_obj.create.return_value = 42
        mock_connection = MagicMock()
        mock_connection.get_model.side_effect = lambda model_name: {
            "res.lang": mock_lang_model,
            "base.language.install": mock_wizard_obj,
        }.get(model_name, MagicMock())
        mock_get_connection.return_value = mock_connection

        run_language_installation(config="dummy.conf", languages=["de_DE", "es_ES"])

        mock_lang_model.search.assert_called_once_with(
            [("code", "in", ["de_DE", "es_ES"])], context={"active_test": False}
        )
        mock_wizard_obj.create.assert_called_once_with(
            {expected_key: [(6, 0, [101, 102])]}
        )
        mock_wizard_obj.browse.assert_called_once_with(42)
        mock_wizard_obj.browse.return_value.lang_install.assert_called_once()
        mock_wait.assert_called_once()

    @patch("odoo_data_flow.lib.actions.language_installer.log.error")
    @patch(
        "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
    )
    def test_run_installation_connection_error(
        self, mock_get_connection: MagicMock, mock_log_error: MagicMock
    ) -> None:
        """Tests that an error is logged if the Odoo connection fails."""
        mock_get_connection.side_effect = Exception("Connection Refused")
        result = run_language_installation(config="bad.conf", languages=["en_US"])
        assert result is False
        mock_log_error.assert_called_once()
        assert "Failed to connect to Odoo" in mock_log_error.call_args[0][0]

    @patch(
        "odoo_data_flow.lib.actions.language_installer._wait_for_languages_to_be_active",
        return_value=False,
    )
    @patch("odoo_data_flow.lib.actions.language_installer.odoo_lib.get_odoo_version")
    @patch(
        "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
    )
    def test_run_installation_wait_fails(
        self,
        mock_get_connection: MagicMock,
        mock_get_odoo_version: MagicMock,
        mock_wait: MagicMock,
    ) -> None:
        """Tests that the main function returns False if the waiting fails."""
        mock_get_odoo_version.return_value = 18
        mock_connection = MagicMock()
        mock_wizard_obj = MagicMock()
        mock_connection.get_model.return_value = mock_wizard_obj
        mock_get_connection.return_value = mock_connection

        result = run_language_installation(config="dummy.conf", languages=["de_DE"])

        assert result is False
        mock_wait.assert_called_once()

    @patch(
        "odoo_data_flow.lib.actions.language_installer._wait_for_languages_to_be_active",
        return_value=True,
    )
    @patch("odoo_data_flow.lib.actions.language_installer.odoo_lib.get_odoo_version")
    @patch(
        "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
    )
    def test_modern_install_no_langs_found(
        self,
        mock_get_connection: MagicMock,
        mock_get_odoo_version: MagicMock,
        mock_wait: MagicMock,
    ) -> None:
        """Tests that a warning is logged if no languages are found to install."""
        mock_get_odoo_version.return_value = 17
        mock_lang_model = MagicMock()
        mock_lang_model.search.return_value = []
        mock_wizard_obj = MagicMock()
        mock_connection = MagicMock()
        mock_connection.get_model.side_effect = lambda model_name: {
            "res.lang": mock_lang_model,
            "base.language.install": mock_wizard_obj,
        }.get(model_name, MagicMock())
        mock_get_connection.return_value = mock_connection

        with patch(
            "odoo_data_flow.lib.actions.language_installer.log.warning"
        ) as mock_log:
            run_language_installation(config="dummy.conf", languages=["xx_XX"])
            mock_log.assert_called_once()
            assert "None of the specified languages" in mock_log.call_args[0][0]
            mock_wizard_obj.create.assert_not_called()

    @patch(
        "odoo_data_flow.lib.actions.language_installer._wait_for_languages_to_be_active",
        return_value=True,
    )
    @patch("odoo_data_flow.lib.actions.language_installer.odoo_lib.get_odoo_version")
    @patch(
        "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
    )
    def test_modern_install_fallback_succeeds(
        self,
        mock_get_connection: MagicMock,
        mock_get_odoo_version: MagicMock,
        mock_wait: MagicMock,
    ) -> None:
        """Tests the fallback from 'lang_ids' to 'langs' when the fallback succeeds."""
        mock_get_odoo_version.return_value = 17
        mock_lang_model = MagicMock()
        mock_lang_model.search.return_value = [101]
        mock_wizard_obj = MagicMock()
        mock_wizard_obj.create.side_effect = [
            Exception("Invalid field lang_ids"),
            42,
        ]
        mock_connection = MagicMock()
        mock_connection.get_model.side_effect = lambda model_name: {
            "res.lang": mock_lang_model,
            "base.language.install": mock_wizard_obj,
        }.get(model_name, MagicMock())
        mock_get_connection.return_value = mock_connection

        run_language_installation(config="dummy.conf", languages=["de_DE"])

        assert mock_wizard_obj.create.call_count == 2
        mock_wizard_obj.create.assert_has_calls(
            [
                call({"lang_ids": [(6, 0, [101])]}),
                call({"langs": [(6, 0, [101])]}),
            ]
        )
        mock_wizard_obj.browse.assert_called_once_with(42)
        mock_wait.assert_called_once()

    @patch(
        "odoo_data_flow.lib.actions.language_installer._wait_for_languages_to_be_active",
        return_value=True,
    )
    @patch("odoo_data_flow.lib.actions.language_installer.odoo_lib.get_odoo_version")
    @patch(
        "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
    )
    def test_modern_install_fallback_fails(
        self,
        mock_get_connection: MagicMock,
        mock_get_odoo_version: MagicMock,
        mock_wait: MagicMock,
    ) -> None:
        """Test Install fallback fails.

        Tests that an error is handled correctly if the fallback create succeeds
        but the subsequent lang_install call fails.
        """
        mock_get_odoo_version.return_value = 17
        mock_lang_model = MagicMock()
        mock_lang_model.search.return_value = [101]
        mock_wizard_obj = MagicMock()
        mock_wizard_obj.create.side_effect = [
            Exception("Invalid field lang_ids"),
            42,
        ]
        mock_wizard_obj.browse.return_value.lang_install.side_effect = Exception(
            "Install failed"
        )

        mock_connection = MagicMock()
        mock_connection.get_model.side_effect = lambda model_name: {
            "res.lang": mock_lang_model,
            "base.language.install": mock_wizard_obj,
        }.get(model_name, MagicMock())
        mock_get_connection.return_value = mock_connection

        with patch(
            "odoo_data_flow.lib.actions.language_installer.log.error"
        ) as mock_log_error:
            result = run_language_installation(config="dummy.conf", languages=["de_DE"])
            assert result is False

            final_log_call = mock_log_error.call_args_list[-1]
            assert "An unexpected error occurred" in final_log_call[0][0]

    @patch(
        "odoo_data_flow.lib.actions.language_installer._wait_for_languages_to_be_active",
        return_value=True,
    )
    @patch("odoo_data_flow.lib.actions.language_installer.odoo_lib.get_odoo_version")
    @patch(
        "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
    )
    def test_modern_install_no_fallback_raises(
        self,
        mock_get_connection: MagicMock,
        mock_get_odoo_version: MagicMock,
        mock_wait: MagicMock,
    ) -> None:
        """Test install modern no fallback.

        Tests that an error is re-raised if the initial create fails and
        there is no fallback.
        """
        mock_get_odoo_version.return_value = 16
        mock_lang_model = MagicMock()
        mock_lang_model.search.return_value = [101]
        mock_wizard_obj = MagicMock()
        mock_wizard_obj.create.side_effect = Exception("Invalid field langs")
        mock_connection = MagicMock()
        mock_connection.get_model.side_effect = lambda model_name: {
            "res.lang": mock_lang_model,
            "base.language.install": mock_wizard_obj,
        }.get(model_name, MagicMock())
        mock_get_connection.return_value = mock_connection

        with patch(
            "odoo_data_flow.lib.actions.language_installer.log.error"
        ) as mock_log_error:
            result = run_language_installation(config="dummy.conf", languages=["de_DE"])
            assert result is False
            final_log_call = mock_log_error.call_args_list[-1]
            assert "An unexpected error occurred" in final_log_call[0][0]

    @patch(
        "odoo_data_flow.lib.actions.language_installer._wait_for_languages_to_be_active",
        return_value=True,
    )
    @patch("odoo_data_flow.lib.actions.language_installer.odoo_lib.get_odoo_version")
    @patch(
        "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
    )
    def test_legacy_install_failure(
        self,
        mock_get_connection: MagicMock,
        mock_get_odoo_version: MagicMock,
        mock_wait: MagicMock,
    ) -> None:
        """Tests error logging during a legacy installation failure."""
        mock_get_odoo_version.return_value = 14
        mock_wizard_obj = MagicMock()
        mock_wizard_obj.create.side_effect = Exception("Create failed")
        mock_connection = MagicMock()
        mock_connection.get_model.return_value = mock_wizard_obj
        mock_get_connection.return_value = mock_connection

        with patch(
            "odoo_data_flow.lib.actions.language_installer.log.error"
        ) as mock_log:
            run_language_installation(config="dummy.conf", languages=["de_DE"])
            mock_log.assert_called_once()
            assert "Failed to install language 'de_DE'" in mock_log.call_args[0][0]

    @patch("odoo_data_flow.lib.actions.language_installer.log.error")
    @patch("odoo_data_flow.lib.actions.language_installer.odoo_lib.get_odoo_version")
    @patch(
        "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
    )
    def test_run_installation_unexpected_error(
        self,
        mock_get_connection: MagicMock,
        mock_get_odoo_version: MagicMock,
        mock_log_error: MagicMock,
    ) -> None:
        """Tests the main exception handler for unexpected errors."""
        mock_get_odoo_version.return_value = 18
        mock_connection = MagicMock()
        mock_connection.get_model.side_effect = Exception("Unexpected Error")
        mock_get_connection.return_value = mock_connection

        result = run_language_installation(config="dummy.conf", languages=["de_DE"])
        assert result is False
        mock_log_error.assert_called_once()
        assert "An unexpected error occurred" in mock_log_error.call_args[0][0]
