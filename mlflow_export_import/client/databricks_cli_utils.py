from mlflow.utils.databricks_utils import get_databricks_host_creds


def get_host_token_for_profile(profile=None):
    """
    :param profile: Databricks profile as in ~/.databrickscfg or None for the default profile
    :return: tuple of (host, token) from the ~/.databrickscfg profile
    """
    server_uri = f"databricks://{profile}" if profile else "databricks"
    creds = get_databricks_host_creds(server_uri)
    return (creds.host, creds.token)


if __name__ == "__main__":
    import sys
    profile = sys.argv[1] if len(sys.argv) > 1 else None
    print("profile:",profile)
    tuple = get_host_token_for_profile(profile)
    print("host and token:", tuple)
