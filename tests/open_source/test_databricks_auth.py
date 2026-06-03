import textwrap
from pathlib import Path
from unittest import mock

import pytest
from mlflow.utils.rest_utils import MlflowHostCreds

from mlflow_export_import.client import databricks_cli_utils


@pytest.fixture
def databricks_config(tmp_path: Path) -> Path:
    content = textwrap.dedent("""
        [DEFAULT]
        host = https://default.azuredatabricks.net
        auth_type = azure-cli

        [explicit]
        host = https://explicit.azuredatabricks.net
        token = dapi_explicit_token

        [azure-cli]
        host = https://azure-cli.azuredatabricks.net
        auth_type = azure-cli
    """).strip()
    config_file = tmp_path / "databrickscfg"
    config_file.write_text(content)
    return config_file


def test_explicit_token_profile(
    databricks_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(databricks_config))

    host, token = databricks_cli_utils.get_host_token_for_profile("explicit")

    assert host == "https://explicit.azuredatabricks.net"
    assert token == "dapi_explicit_token"

@pytest.mark.parametrize(("profile", "expected_uri"), [
    ("azure-cli", "databricks://azure-cli"),
    (None, "databricks"),
])
def test_auth_type_profile(
    databricks_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str | None,
    expected_uri: str,
) -> None:
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(databricks_config))
    mock_creds = mock.Mock(spec=MlflowHostCreds, host="https://resolved.net", token="resolved_token")

    with mock.patch(
        "mlflow_export_import.client.databricks_cli_utils.get_databricks_host_creds",
        return_value=mock_creds,
    ) as mock_fn:
        host, token = databricks_cli_utils.get_host_token_for_profile(profile)

    mock_fn.assert_called_once_with(expected_uri)
    assert (host, token) == ("https://resolved.net", "resolved_token")
